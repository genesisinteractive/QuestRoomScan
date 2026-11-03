using System;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    internal struct UnwrappedMeshResult
    {
        public Vector3[] Positions;
        public Vector3[] Normals;
        public Vector2[] UVs;
        public float[] RawUVs;
        public int[] Indices;
        public int AtlasWidth;
        public int AtlasHeight;
        internal Vector3[] OrigPositions;
        internal Vector3[] OrigNormals;
        internal int[] OrigIndices;
        /// <summary>xatlas chart per output vertex, or null when the unwrap was
        /// reconstructed from disk. Lets the bake prefer one view per chart.</summary>
        internal int[] ChartIndices;
        internal int ChartCount;
    }

    internal struct RefinedTextureResult
    {
        public Vector3[] Positions;
        public Vector3[] Normals;
        public Vector2[] UVs;
        public int[] Indices;
        public byte[] AtlasPixels;  // RGBA32
        public byte[] NormalPixels; // RGBA32 Sobel normal map (may be null for older packages / CPU bake)
        public int AtlasWidth;
        public int AtlasHeight;
    }

    /// <summary>
    /// On-device texture refinement pipeline: reads back the GPU mesh,
    /// UV-unwraps via xatlas, and bakes a sharp texture atlas from saved keyframes.
    /// All CPU-heavy work runs on background threads.
    /// Attach as an optional module on the same GameObject as <see cref="RoomScanner"/>.
    /// </summary>
    [RequireComponent(typeof(KeyframeCollector))]
    public class TextureRefinement : MonoBehaviour, IRoomScanModule
    {
        public string ModuleName => "Texture Refinement";

        [Header("Bake Pipeline")]
        [SerializeField] internal Shader refinedMeshShader;
        [SerializeField] internal Shader refinedMeshBackfaceShader;
        [SerializeField] internal Shader occlusionMeshShader;
        [SerializeField] internal ComputeShader atlasBakeCompute;
        [Tooltip("Force CPU bake path instead of GPU compute")]
        [SerializeField] internal bool forceCpuBake = false;
        [Tooltip("Skip denoise pass after baking")]
        [SerializeField] internal bool skipDenoise = true;
        [Tooltip("Multi-view blend: 2-pass GPU bake that blends top views per texel")]
        [SerializeField] internal bool multiViewBlend = true;
        [Tooltip("Unsharp mask strength (0 = off)")]
        [Range(0f, 2f)]
        [SerializeField] internal float sharpenStrength = 0.8f;
        [Tooltip("Sharpening kernel radius")]
        [Range(1, 4)]
        [SerializeField] internal int sharpenRadius = 2;
        [Tooltip("Level colour across UV seams: the two atlas sides of every seam edge are pulled to their mean and the correction is diffused into each chart, so a view switch at a chart border becomes a gradient instead of a step.")]
        [SerializeField] internal bool enableSeamBlending = true;
        [Tooltip("Diffusion steps for the seam correction; each walks ~2 texels into the chart. 40 ≈ 60-80 texel feather. One GPU pass per step, spread over frames.")]
        [Range(4, 96)]
        [SerializeField] internal int seamLevelIterations = 40;
        [Tooltip("Start of the blend admission ramp as a fraction of the texel's best *hero* score. Heroes are head-on views in the working-distance band; close-ups fill holes in pass 1 but do not enter this ramp.")]
        [Range(0.1f, 0.95f)]
        [SerializeField] internal float blendMinFraction = 0.75f;
        [Tooltip("At most this many hero views blended per texel. Pass 2 walks keyframes best-score-first so the cap keeps the top-K, not capture order.")]
        [Range(1, 8)]
        [SerializeField] internal int maxViewsPerTexel = 3;
        [Tooltip("Minimum N·V for a view to count as head-on (hero / blend). Grazing close-ups fail this even if they won the old 1/distance score.")]
        [Range(0.15f, 0.85f)]
        [SerializeField] internal float blendFacingMin = 0.45f;
        [Tooltip("A keyframe may be the chart's preferred view only if its hero samples cover this fraction of the chart. Stops a close-up of a corner from locking the whole island.")]
        [Range(0f, 0.6f)]
        [SerializeField] internal float chartMinCover = 0.15f;
        [Tooltip("Sobel normal map strength (0 = skip normal map generation)")]
        [Range(0f, 20f)]
        [SerializeField] internal float normalStrength = 8f;
        [Header("HQ Server Refinement")]
        [Tooltip("Server-side atlas super-resolution scale")]
        [Range(1, 4)]
        [SerializeField] internal int hqRefineScale = 2;

        [Header("Unwrap")]
        [Tooltip("Align charts to 4x4 blocks for faster packing")]
        [SerializeField] internal bool useBlockAlign = true;
        [Tooltip("Chart growth cost limit")]
        [Range(0.5f, 4f)]
        [SerializeField] internal float xatlasMaxCost = 1.5f;
        [Tooltip("Threads xatlas may use for the unwrap, including the calling worker (0 = every core). xatlas defaults to all cores; 3 leaves the rest to the frame.")]
        [Range(0, 8)]
        [SerializeField] internal int xatlasThreads = 3;
        [Tooltip("POSIX nice for the unwrap threads on Android (0 = normal, 19 = lowest). With a positive value the engine's threads win every contended core.")]
        [Range(0, 19)]
        [SerializeField] internal int xatlasThreadNice = 10;

        [Header("Simplification")]
        [Tooltip("Target triangle ratio for the refined mesh (1.0 = disabled, 0.5 = 50% triangles). Applied before the unwrap or after the bake, see below. Runs on a background thread.")]
        [Range(0.1f, 1f)]
        [SerializeField] internal float postBakeSimplificationRatio = 0.5f;
        [Tooltip("On: simplify geometry before the UV unwrap (meshopt_simplify) so xatlas and both bake passes run on the reduced mesh. The dense mesh still builds occlusion depth. Off: unwrap and bake the dense mesh, then simplify with UV-locked borders.")]
        [SerializeField] internal bool simplifyBeforeUnwrap = true;

        [Header("Keyframe Registration")]
        [Tooltip("After the first bake pass, align each keyframe to the atlas rendered from its own " +
                 "pose (low-res ZNCC image shift → small rotation) before the blend pass. Absorbs " +
                 "exposure-time pose error and residual mesh bias. One extra light GPU pass per keyframe.")]
        [SerializeField] internal bool refineKeyframePoses = true;
        [Tooltip("Search radius in low-res pixels (image downsampled by 4). 6 ≈ ±1.6° at Quest 3 focal length.")]
        [Range(2, 12)]
        [SerializeField] internal int registrationSearchRadius = 6;
        [Tooltip("Minimum normalized cross-correlation at the best shift to accept a correction.")]
        [Range(0.05f, 0.9f)]
        [SerializeField] internal float registrationMinNcc = 0.25f;

        [Header("Motion Blur")]
        [Tooltip("Weigh each keyframe's view score by how still the head was when it was taken, so a steady view of a " +
                 "surface wins over a smeared one taken during a fast turn, and a smeared one only fills what no steady " +
                 "view saw. Speed comes from the collector's \"av\" (deg/s at capture), or for older packages from the " +
                 "neighbouring keyframes' poses.")]
        [SerializeField] internal bool weighBySharpness = true;
        [Tooltip("Head turn speed (deg/s) at which a keyframe counts half. At a 10 ms exposure and Quest 3 focal " +
                 "length, 25 deg/s is a smear of about 4 px; 60 deg/s about 9 px.")]
        [Range(5f, 120f)]
        [SerializeField] internal float sharpHalfDegPerSec = 25f;
        [Tooltip("Lowest weight a fast keyframe falls to: it still fills a surface no steadier view saw.")]
        [Range(0.01f, 1f)]
        [SerializeField] internal float sharpMinWeight = 0.05f;

        [Header("Exposure Equalisation")]
        [Tooltip("With registration on, measure each keyframe's colour against the pass-1 atlas (now the working-distance winner) and scale the photo to match before blending. Close-up auto-exposure no longer sets room brightness.")]
        [SerializeField] internal bool equalizeExposure = true;
        [Tooltip("Gain is clamped to [1/limit, limit] per channel. 1.6 covers a stop of auto-exposure; more and a view that saw mostly a lamp would darken a whole wall.")]
        [Range(1.1f, 3f)]
        [SerializeField] internal float exposureGainLimit = 1.6f;

        [Header("Chart-Consistent Blend")]
        [Tooltip("Weight multiplier for the working-distance view that covers enough of a UV chart. " +
                 "View switches move to chart borders (which the seam pass already blends). 1 = off.")]
        [Range(1f, 8f)]
        [SerializeField] internal float chartBestViewBoost = 3f;

        [Tooltip("The per-keyframe occlusion depth (dense mesh, one atomic per covered photo pixel) is rasterised at photo resolution divided by this. A 5 cm depth tolerance does not need 1280×960; 2 quarters the heaviest raster of the bake. 1 = full resolution.")]
        [Range(1, 4)]
        [SerializeField] internal int occlusionDepthDivisor = 2;

        [Header("Diagnostics")]
        [Tooltip("Write one [TextureRefine][Profile] block per refinement: wall time per stage, main-thread and worker time per keyframe, and the compositor frame times the bake ran across (count, max, missed 72 Hz, hitches). Off by default.")]
        [SerializeField] internal bool profileRefinement = false;

        private RoomScanner _scanner;
        private RefineProfile _profile;

        /// <summary>One frame later, counted by the profile.</summary>
        async Task NextFrame()
        {
            await Task.Yield();
            _profile?.Frame();
        }

        /// <summary>
        /// Await a worker task while sampling every compositor frame it spans,
        /// so a long native stage (xatlas, meshopt) shows in the profile as the
        /// frames it cost the player rather than as a gap.
        /// </summary>
        async Task AwaitSampled(Task work)
        {
            if (_profile == null) { await work; return; }
            while (!work.IsCompleted)
                await NextFrame();
            await work;   // rethrow
        }

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _scanner = scanner;
        }

        /// <summary>Keyframe reads + decodes kept in flight ahead of the bake.</summary>
        const int DecodePrefetchDepth = 3;

        // Working-distance band (metres). Peak 0.8–2 m; strong penalty below
        // 0.45 m (macro / AE / perspective); mild falloff past 3.5 m.
        internal const float ViewCloseM = 0.45f;
        internal const float ViewPeak0 = 0.8f;
        internal const float ViewPeak1 = 2.0f;
        internal const float ViewFarM = 3.5f;

        internal static float ViewDistanceWeight(float d)
        {
            if (d < ViewCloseM) return 0.08f + 0.27f * Mathf.Clamp01(d / ViewCloseM);
            if (d < ViewPeak0) return Mathf.Lerp(0.35f, 1f, (d - ViewCloseM) / (ViewPeak0 - ViewCloseM));
            if (d <= ViewPeak1) return 1f;
            if (d < ViewFarM) return Mathf.Lerp(1f, 0.45f, (d - ViewPeak1) / (ViewFarM - ViewPeak1));
            return 0.45f * ViewFarM / d;
        }

        internal static float ViewGsdWeight(float d, float fx)
        {
            float pxm = d / Mathf.Max(fx, 1f);
            float t = (pxm - 0.002f) / 0.0025f;
            return Mathf.Max(Mathf.Exp(-0.5f * t * t), 0.25f);
        }

        /// <summary>Head-on × working-distance × GSD. Positive for close-ups so they still fill holes.</summary>
        internal static float ViewScore(float dotNV, float distM, float fx)
            => (dotNV * dotNV) * ViewDistanceWeight(distM) * ViewGsdWeight(distM, fx);

        void BindViewSelection(ComputeShader compute)
        {
            compute.SetFloat("_BlendFacingMin", blendFacingMin);
            compute.SetFloat("_ViewCloseM", ViewCloseM);
            compute.SetFloat("_ViewPeak0", ViewPeak0);
            compute.SetFloat("_ViewPeak1", ViewPeak1);
            compute.SetFloat("_ViewFarM", ViewFarM);
            compute.SetFloat("_ChartMinCover", chartMinCover);
        }

        // pos @ 0 is the extractor dump. prevPos @ 12 is presentation-only.
        const int VertStride = GPUSurfaceNets.VertexStride;
        const int VertPos = GPUSurfaceNets.VertexPosOffset;
        const int VertNorm = GPUSurfaceNets.VertexNormalOffset;
        const int VertPacked = GPUSurfaceNets.VertexPackedColorOffset;

        internal event Action<string> StatusChanged;

        // ═══════════════════════════════════════════════════════════════
        //  UV UNWRAP (shared prerequisite for both on-device and HQ refine)
        // ═══════════════════════════════════════════════════════════════

        internal Task<UnwrappedMeshResult> UnwrapMeshAsync(
            string keyframeDir, Matrix4x4 keyframeRelocation, int atlasResolution = 2048)
        {
            var opts = XAtlasWrapper.UnwrapOptions.Default;
            opts.Resolution = (uint)atlasResolution;
            opts.MaxCost = xatlasMaxCost;
            opts.BlockAlign = useBlockAlign;
            opts.MaxThreads = xatlasThreads;
            opts.WorkerNice = xatlasThreadNice;
            return UnwrapMeshAsync(keyframeDir, keyframeRelocation, opts);
        }

        internal async Task<UnwrappedMeshResult> UnwrapMeshAsync(
            string keyframeDir, Matrix4x4 keyframeRelocation,
            XAtlasWrapper.UnwrapOptions opts)
        {
            _profile = profileRefinement ? new RefineProfile() : null;
            var readScope = _profile?.Stage("meshReadback");
            ReportStatus("Reading mesh from GPU...");
            MeshExtractor.Instance?.ExtractForAuthoring();
            var (positions, normals, colors, indices) = await ReadbackMeshAsync();
            _profile?.Frame();   // the extraction + readback frame lands on this stage, not the next
            readScope?.Dispose();
            if (positions == null || positions.Length == 0)
                throw new InvalidOperationException("Mesh readback returned no vertices");

            Logger.Info($"[TextureRefine] Readback: {positions.Length} verts, {indices.Length / 3} tris");

            Vector3[] inPos = positions;
            Vector3[] inNorm = normals;
            int[] inIdx = indices;

            if (simplifyBeforeUnwrap && postBakeSimplificationRatio < 1f)
            {
                using var _ = _profile?.Stage("simplify") ?? default;
                float ratio = Mathf.Clamp(postBakeSimplificationRatio, 0.05f, 1f);
                ReportStatus($"Simplifying mesh ({ratio:P0})...");
                Vector3[] sPos = null, sNorm = null;
                int[] sIdx = null;
                bool ok = false;
                await AwaitSampled(Task.Run(() =>
                {
                    try
                    {
                        ok = XAtlasWrapper.SimplifyGeometry(positions, normals, indices, ratio, out sPos, out sNorm, out sIdx);
                    }
                    catch (Exception e)
                    {
                        Logger.Warning($"[TextureRefine] Pre-unwrap simplify unavailable ({e.Message}); unwrapping the dense mesh.");
                    }
                }));
                if (ok)
                {
                    inPos = sPos;
                    inNorm = sNorm;
                    inIdx = sIdx;
                }
            }

            var unwrapScope = _profile?.Stage("xatlas");
            ReportStatus("UV unwrapping...");
            XAtlasWrapper.Result uvResult = default;
            Vector3[] outPos = null;
            Vector3[] outNorm = null;
            Vector2[] outUVs = null;

            await AwaitSampled(Task.Run(() =>
            {
                float[] flatPos = new float[inPos.Length * 3];
                float[] flatNorm = new float[inNorm.Length * 3];
                for (int i = 0; i < inPos.Length; i++)
                {
                    flatPos[i * 3] = inPos[i].x;
                    flatPos[i * 3 + 1] = inPos[i].y;
                    flatPos[i * 3 + 2] = inPos[i].z;
                    flatNorm[i * 3] = inNorm[i].x;
                    flatNorm[i * 3 + 1] = inNorm[i].y;
                    flatNorm[i * 3 + 2] = inNorm[i].z;
                }
                uvResult = XAtlasWrapper.Unwrap(flatPos, flatNorm, inPos.Length,
                    inIdx, inIdx.Length, opts);
                if (uvResult.VertexCount == 0) return;

                int aw = uvResult.AtlasWidth;
                int ah = uvResult.AtlasHeight;
                outPos = new Vector3[uvResult.VertexCount];
                outNorm = new Vector3[uvResult.VertexCount];
                outUVs = new Vector2[uvResult.VertexCount];
                for (int i = 0; i < uvResult.VertexCount; i++)
                {
                    int src = uvResult.Xrefs[i];
                    outPos[i] = inPos[src];
                    outNorm[i] = inNorm[src];
                    outUVs[i] = new Vector2(
                        uvResult.UVs[i * 2] / aw,
                        uvResult.UVs[i * 2 + 1] / ah);
                }
            }));

            if (uvResult.VertexCount == 0 || outPos == null)
                throw new InvalidOperationException("xatlas produced no output vertices");

            int atlasW = uvResult.AtlasWidth;
            int atlasH = uvResult.AtlasHeight;
            Logger.Info($"[TextureRefine] xatlas: {uvResult.VertexCount} verts, " +
                      $"{uvResult.IndexCount / 3} tris, atlas {atlasW}x{atlasH}");
            unwrapScope?.Dispose();
            await NextFrame();

            return new UnwrappedMeshResult
            {
                Positions = outPos,
                Normals = outNorm,
                UVs = outUVs,
                RawUVs = uvResult.UVs,
                Indices = uvResult.Indices,
                AtlasWidth = atlasW,
                AtlasHeight = atlasH,
                OrigPositions = positions,
                OrigNormals = normals,
                OrigIndices = indices,
                ChartIndices = uvResult.ChartIndices,
                ChartCount = uvResult.ChartCount,
            };
        }

        // ═══════════════════════════════════════════════════════════════
        //  MESH READBACK
        // ═══════════════════════════════════════════════════════════════

        /// <summary>Read back the first <paramref name="byteCount"/> bytes (0 = whole buffer).
        /// The extractor's buffers are sized to a voxel budget, tens of MB; the
        /// mesh in them is a few MB. Reading the capacity was a 100 ms frame.</summary>
        static Task<byte[]> ReadbackBytesAsync(GraphicsBuffer buffer, int byteCount = 0)
        {
            var tcs = new System.Threading.Tasks.TaskCompletionSource<byte[]>();
            System.Action<AsyncGPUReadbackRequest> onDone = request =>
            {
                if (request.hasError) { tcs.SetResult(null); return; }
                var native = request.GetData<byte>();
                byte[] managed = new byte[native.Length];
                NativeArray<byte>.Copy(native, managed, native.Length);
                tcs.SetResult(managed);
            };
            if (byteCount > 0)
                AsyncGPUReadback.Request(buffer, byteCount, 0, onDone);
            else
                AsyncGPUReadback.Request(buffer, onDone);
            return tcs.Task;
        }

        static async Task<(Vector3[], Vector3[], Color32[], int[])> ReadbackMeshAsync()
        {
            var gpuSN = MeshExtractor.Instance?.GpuSurfaceNets;
            if (gpuSN == null || gpuSN.VertexBuffer == null || gpuSN.IndexBuffer == null)
            {
                Logger.Error("[TextureRefine] GpuSurfaceNets or its buffers are null");
                return (null, null, null, null);
            }

            Logger.Info("[TextureRefine] Starting GPU readback...");

            // Read all three buffers using callback-based readback
            // (copies NativeArray to managed array immediately in callback frame)
            byte[] counterBytes = await ReadbackBytesAsync(gpuSN.CountersBuffer);
            if (counterBytes == null)
            {
                Logger.Error("[TextureRefine] Counter readback failed");
                return (null, null, null, null);
            }

            int vertCount = BitConverter.ToInt32(counterBytes, 0);
            int idxCount = counterBytes.Length >= 8 ? BitConverter.ToInt32(counterBytes, 4) : 0;

            Logger.Info($"[TextureRefine] Counters: verts={vertCount}, idx={idxCount}");

            if (vertCount <= 0 || idxCount <= 0)
            {
                Logger.Warning($"[TextureRefine] No mesh data: verts={vertCount}, idx={idxCount}");
                return (null, null, null, null);
            }

            vertCount = Mathf.Min(vertCount, gpuSN.VertexBuffer.count);
            idxCount = Mathf.Min(idxCount, gpuSN.IndexBuffer.count);

            var vertTask = ReadbackBytesAsync(gpuSN.VertexBuffer, vertCount * VertStride);
            var idxTask = ReadbackBytesAsync(gpuSN.IndexBuffer, idxCount * 4);
            byte[] vertData = await vertTask;
            byte[] idxData = await idxTask;
            if (vertData == null || idxData == null)
            {
                Logger.Error("[TextureRefine] Mesh readback failed");
                return (null, null, null, null);
            }

            // Parse on a worker: a few MB of BitConverter is still a frame's worth.
            Vector3[] positions = null, normals = null;
            Color32[] colors = null;
            int[] indices = null;
            await Task.Run(() =>
            {
                int vc = Mathf.Min(vertCount, vertData.Length / VertStride);
                int ic = Mathf.Min(idxCount, idxData.Length / 4);
                indices = new int[ic];
                Buffer.BlockCopy(idxData, 0, indices, 0, ic * 4);

                positions = new Vector3[vc];
                normals = new Vector3[vc];
                colors = new Color32[vc];
                for (int i = 0; i < vc; i++)
                {
                    int off = i * VertStride;
                    positions[i] = new Vector3(
                        BitConverter.ToSingle(vertData, off + VertPos),
                        BitConverter.ToSingle(vertData, off + VertPos + 4),
                        BitConverter.ToSingle(vertData, off + VertPos + 8));
                    normals[i] = new Vector3(
                        BitConverter.ToSingle(vertData, off + VertNorm),
                        BitConverter.ToSingle(vertData, off + VertNorm + 4),
                        BitConverter.ToSingle(vertData, off + VertNorm + 8));
                    uint packed = BitConverter.ToUInt32(vertData, off + VertPacked);
                    colors[i] = new Color32(
                        (byte)(packed & 0xFF),
                        (byte)((packed >> 8) & 0xFF),
                        (byte)((packed >> 16) & 0xFF),
                        255);
                }
            });

            Logger.Info($"[TextureRefine] Readback complete: {positions.Length} verts, {indices.Length / 3} tris");
            return (positions, normals, colors, indices);
        }

        // ═══════════════════════════════════════════════════════════════
        //  TEXTURE BAKE
        // ═══════════════════════════════════════════════════════════════

        struct Keyframe
        {
            public byte[] Pixels; // RGBA32, row-major (or compressed JPEG before decode)
            public int Width, Height;
            public int SensorWidth, SensorHeight;
            public Vector3 Position;
            public Quaternion Rotation;
            public float Fx, Fy, Cx, Cy;
            public string JpgPath; // deferred: path to JPEG, read on demand to avoid OOM
            /// <summary>Hand / forearm capsules at capture (xyz + radius per end), or null.</summary>
            public Vector4[] BodyCapP0, BodyCapP1;
            /// <summary>Median scene distance at capture (m), 0 = unknown. Written as "z" in frames.jsonl.</summary>
            public float SceneZ;
            /// <summary>RGB exposure gain toward the pass-1 atlas, set by registration; zero = unset (1).</summary>
            public Vector3 Gain;
            public Vector4 GainOrOne => Gain == Vector3.zero ? Vector4.one : new Vector4(Gain.x, Gain.y, Gain.z, 1f);
            /// <summary>Capture time (s, "ts").</summary>
            public float Ts;
            /// <summary>Head turn speed at capture (deg/s, "av"); negative = not recorded.</summary>
            public float AngVel;
            /// <summary>View-score multiplier from <see cref="ApplySharpnessWeights"/>; zero = unset (1).</summary>
            public float Weight;
            public float WeightOrOne => Weight > 0f ? Weight : 1f;
        }

        const int BodyCapMax = 8;
        static readonly Vector4[] s_bodyCapP0 = new Vector4[BodyCapMax];
        static readonly Vector4[] s_bodyCapP1 = new Vector4[BodyCapMax];

        /// <summary>
        /// Upload this view's hand capsules so the bake skips texels whose line
        /// of sight passed through a hand. Radius margin covers the blurry
        /// edge and the controller.
        /// </summary>
        static void BindBodyCapsules(ComputeShader compute, in Keyframe kf)
        {
            int n = kf.BodyCapP0 != null ? Mathf.Min(kf.BodyCapP0.Length, BodyCapMax) : 0;
            for (int i = 0; i < BodyCapMax; i++)
            {
                s_bodyCapP0[i] = i < n ? kf.BodyCapP0[i] : Vector4.zero;
                s_bodyCapP1[i] = i < n ? kf.BodyCapP1[i] : Vector4.zero;
            }
            compute.SetInt("_BodyCapCount", n);
            compute.SetVectorArray("_BodyCapP0", s_bodyCapP0);
            compute.SetVectorArray("_BodyCapP1", s_bodyCapP1);
            compute.SetFloat("_BodyCapMargin", 1.4f);
        }

        struct TriData
        {
            public int I0, I1, I2;
            public Vector3 FaceNormal;
            public Vector3 Centroid;
            public float U0, V0, U1, V1, U2, V2;
        }

        static TriData[] PrecomputeTriData(
            Vector3[] outPos, Vector3[] outNorm, float[] rawUVs, int[] indices, int outVertCount)
        {
            int triCount = indices.Length / 3;
            var data = new TriData[triCount];
            for (int t = 0; t < triCount; t++)
            {
                int i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
                if (i0 >= outVertCount || i1 >= outVertCount || i2 >= outVertCount)
                {
                    data[t].I0 = -1;
                    continue;
                }

                Vector3 p0 = outPos[i0], p1 = outPos[i1], p2 = outPos[i2];
                Vector3 fn = Vector3.Cross(p1 - p0, p2 - p0).normalized;
                if (fn.sqrMagnitude < 0.001f) fn = outNorm[i0];

                data[t].I0 = i0; data[t].I1 = i1; data[t].I2 = i2;
                data[t].FaceNormal = fn;
                data[t].Centroid = (p0 + p1 + p2) / 3f;
                data[t].U0 = rawUVs[i0 * 2]; data[t].V0 = rawUVs[i0 * 2 + 1];
                data[t].U1 = rawUVs[i1 * 2]; data[t].V1 = rawUVs[i1 * 2 + 1];
                data[t].U2 = rawUVs[i2 * 2]; data[t].V2 = rawUVs[i2 * 2 + 1];
            }
            return data;
        }

        /// <summary>
        /// Returns the contents of frames.jsonl with poses transformed by the relocation matrix.
        /// Used by GS training and HQ refine upload to send corrected poses to the server.
        /// </summary>
        public static byte[] RelocateFramesJsonl(string keyframeDir, Matrix4x4 relocation)
        {
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");
            if (!File.Exists(manifestPath)) return null;

            if (relocation == Matrix4x4.identity)
                return File.ReadAllBytes(manifestPath);

            var ci = System.Globalization.CultureInfo.InvariantCulture;
            string[] lines = File.ReadAllLines(manifestPath);
            var sb = new System.Text.StringBuilder(lines.Length * 256);
            var rot = relocation.rotation;

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var kf = ParseKeyframe(line, "");
                    var pos = relocation.MultiplyPoint3x4(kf.Position);
                    var q = rot * kf.Rotation;

                    string relocated = System.Text.RegularExpressions.Regex.Replace(line,
                        "\"px\":[^,}]+", $"\"px\":{pos.x.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"py\":[^,}]+", $"\"py\":{pos.y.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"pz\":[^,}]+", $"\"pz\":{pos.z.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qx\":[^,}]+", $"\"qx\":{q.x.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qy\":[^,}]+", $"\"qy\":{q.y.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qz\":[^,}]+", $"\"qz\":{q.z.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qw\":[^,}]+", $"\"qw\":{q.w.ToString("F6", ci)}");

                    sb.AppendLine(relocated);
                }
                catch { sb.AppendLine(line); }
            }
            return System.Text.Encoding.UTF8.GetBytes(sb.ToString());
        }

        static System.Collections.Generic.List<Keyframe> ParseKeyframeManifest(
            string keyframeDir, Matrix4x4 keyframeRelocation)
        {
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");
            string imagesDir = Path.Combine(keyframeDir, "images");
            var metaList = new System.Collections.Generic.List<Keyframe>();

            if (!File.Exists(manifestPath)) return metaList;

            string[] lines = File.ReadAllLines(manifestPath);
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var kf = ParseKeyframe(line, imagesDir);
                    if (string.IsNullOrEmpty(kf.JpgPath)) continue;
                    if (keyframeRelocation != Matrix4x4.identity)
                    {
                        kf.Position = keyframeRelocation.MultiplyPoint3x4(kf.Position);
                        kf.Rotation = keyframeRelocation.rotation * kf.Rotation;
                        if (kf.BodyCapP0 != null)
                        {
                            for (int i = 0; i < kf.BodyCapP0.Length; i++)
                            {
                                Vector3 a = keyframeRelocation.MultiplyPoint3x4((Vector3)kf.BodyCapP0[i]);
                                Vector3 b = keyframeRelocation.MultiplyPoint3x4((Vector3)kf.BodyCapP1[i]);
                                kf.BodyCapP0[i] = new Vector4(a.x, a.y, a.z, kf.BodyCapP0[i].w);
                                kf.BodyCapP1[i] = new Vector4(b.x, b.y, b.z, kf.BodyCapP1[i].w);
                            }
                        }
                    }
                    metaList.Add(kf);
                }
                catch (Exception e)
                {
                    Logger.Warning($"[TextureRefine] Skip keyframe: {e.Message}");
                }
            }
            return metaList;
        }

        /// <summary>
        /// Set each keyframe's <see cref="Keyframe.Weight"/> from how fast the head turned when it was taken:
        /// <c>1 / (1 + (speed / sharpHalfDegPerSec)²)</c>, floored at <see cref="sharpMinWeight"/>. The speed is the
        /// collector's own measurement at capture ("av") when the package has it; otherwise the turn to the
        /// neighbouring keyframes over the time between them (the larger side, neighbours within a second only), an
        /// average that reads a short fast flick as slower than it was. No speed at all keeps weight 1.
        /// </summary>
        void ApplySharpnessWeights(System.Collections.Generic.List<Keyframe> list)
        {
            if (!weighBySharpness || list.Count == 0) return;
            int fromCapture = 0, fromNeighbours = 0, unknown = 0, low = 0;
            var speeds = new System.Collections.Generic.List<float>(list.Count);
            var weights = new System.Collections.Generic.List<float>(list.Count);
            float Speed(in Keyframe a, in Keyframe b)
            {
                float dt = Mathf.Abs(a.Ts - b.Ts);
                return dt < 1e-3f || dt > 1f ? -1f : Quaternion.Angle(a.Rotation, b.Rotation) / dt;
            }
            for (int i = 0; i < list.Count; i++)
            {
                var kf = list[i];
                float speed = kf.AngVel;
                if (speed >= 0f) fromCapture++;
                else
                {
                    if (i > 0) speed = Mathf.Max(speed, Speed(kf, list[i - 1]));
                    if (i + 1 < list.Count) speed = Mathf.Max(speed, Speed(kf, list[i + 1]));
                    if (speed >= 0f) fromNeighbours++;
                }
                if (speed < 0f)
                {
                    unknown++;
                    kf.Weight = 1f;
                }
                else
                {
                    float r = speed / Mathf.Max(1f, sharpHalfDegPerSec);
                    kf.Weight = Mathf.Max(sharpMinWeight, 1f / (1f + r * r));
                    speeds.Add(speed);
                }
                if (kf.Weight < 0.25f) low++;
                weights.Add(kf.Weight);
                list[i] = kf;
            }
            speeds.Sort();
            weights.Sort();
            float Median(System.Collections.Generic.List<float> v) => v.Count == 0 ? 0f : v[v.Count / 2];
            Logger.Info($"[TextureRefine] Sharpness: {list.Count} keyframes, head speed median {Median(speeds):F0} deg/s " +
                        $"(at capture {fromCapture}, from neighbours {fromNeighbours}, unknown {unknown}); weight median " +
                        $"{Median(weights):F2}, {low} below 0.25 (half at {sharpHalfDegPerSec:F0} deg/s, floor {sharpMinWeight:F2})");
        }

        /// <summary>"x y z x y z r;..." as written by the keyframe collector.</summary>
        static void ParseBodyCapsules(string caps, ref Keyframe kf)
        {
            if (string.IsNullOrEmpty(caps)) return;
            var ci = System.Globalization.CultureInfo.InvariantCulture;
            string[] entries = caps.Split(';');
            var p0 = new System.Collections.Generic.List<Vector4>(entries.Length);
            var p1 = new System.Collections.Generic.List<Vector4>(entries.Length);
            foreach (string e in entries)
            {
                string[] f = e.Split(' ');
                if (f.Length < 7) continue;
                if (!float.TryParse(f[0], System.Globalization.NumberStyles.Float, ci, out float ax)) continue;
                if (!float.TryParse(f[1], System.Globalization.NumberStyles.Float, ci, out float ay)) continue;
                if (!float.TryParse(f[2], System.Globalization.NumberStyles.Float, ci, out float az)) continue;
                if (!float.TryParse(f[3], System.Globalization.NumberStyles.Float, ci, out float bx)) continue;
                if (!float.TryParse(f[4], System.Globalization.NumberStyles.Float, ci, out float by)) continue;
                if (!float.TryParse(f[5], System.Globalization.NumberStyles.Float, ci, out float bz)) continue;
                if (!float.TryParse(f[6], System.Globalization.NumberStyles.Float, ci, out float r)) continue;
                p0.Add(new Vector4(ax, ay, az, r));
                p1.Add(new Vector4(bx, by, bz, r));
                if (p0.Count >= BodyCapMax) break;
            }
            if (p0.Count == 0) return;
            kf.BodyCapP0 = p0.ToArray();
            kf.BodyCapP1 = p1.ToArray();
        }

        static Keyframe ParseKeyframe(string jsonLine, string imagesDir)
        {
            var kf = new Keyframe();
            // Minimal JSON parsing without dependency
            float px = 0, py = 0, pz = 0, qx = 0, qy = 0, qz = 0, qw = 1;
            int id = 0;
            float fx = 0, fy = 0, cx = 0, cy = 0;
            int sw = 0, sh = 0;
            string caps = null;
            float sceneZ = 0;
            float ts = 0, av = -1;

            foreach (string token in jsonLine.Trim('{', '}', ' ').Split(','))
            {
                string[] kv = token.Split(':');
                if (kv.Length < 2) continue;
                string key = kv[0].Trim('"', ' ');
                string val = kv[1].Trim('"', ' ');
                switch (key)
                {
                    case "cap": caps = val; break;
                    case "z": sceneZ = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "id": id = int.Parse(val); break;
                    case "ts": ts = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "av": av = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "px": px = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "py": py = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "pz": pz = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qx": qx = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qy": qy = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qz": qz = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qw": qw = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "fx": fx = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "fy": fy = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "cx": cx = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "cy": cy = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "sw": sw = int.Parse(val); break;
                    case "sh": sh = int.Parse(val); break;
                }
            }

            kf.Position = new Vector3(px, py, pz);
            kf.Rotation = new Quaternion(qx, qy, qz, qw);
            kf.Fx = fx; kf.Fy = fy;
            kf.Cx = cx; kf.Cy = cy;
            kf.SensorWidth = sw;
            kf.SensorHeight = sh;
            kf.SceneZ = sceneZ;
            kf.Ts = ts;
            kf.AngVel = av;
            ParseBodyCapsules(caps, ref kf);

            string imgPath = Path.Combine(imagesDir, $"{id:D6}.jpg");
            if (!File.Exists(imgPath)) return kf;

            // Store path only — JPEG bytes read on-demand during bake to avoid OOM with many keyframes
            kf.JpgPath = imgPath;
            kf.Pixels = new byte[0]; // non-null signals "has image"
            return kf;
        }

        /// <summary>
        /// GPU-accelerated atlas baking via compute shader.
        /// Mirrors the CPU rasterization logic 1:1 using integer pixel indexing
        /// on StructuredBuffers — no texture UV ambiguity, no framebuffer orientation issues.
        /// Falls back to CPU path if the compute shader is unavailable.
        /// </summary>
        internal async Task<(byte[] atlas, byte[] normal)> BakeAtlasAsync(
            UnwrappedMeshResult mesh, string keyframeDir, Matrix4x4 keyframeRelocation)
        {
            ComputeShader compute = forceCpuBake ? null : atlasBakeCompute;
            if (compute == null)
            {
                Logger.Warning("[TextureRefine] No compute shader, falling back to CPU bake");
                return (await BakeAtlasCPUAsync(mesh, keyframeDir, keyframeRelocation), null);
            }

            if (profileRefinement) _profile ??= new RefineProfile();
            var manifestScope = _profile?.Stage("manifest");
            ReportStatus("Loading keyframe metadata...");
            var relocation = keyframeRelocation;
            var metaList = await Task.Run(() => ParseKeyframeManifest(keyframeDir, relocation));
            if (metaList.Count == 0)
                throw new InvalidOperationException("No keyframes available for baking");
            ApplySharpnessWeights(metaList);
            manifestScope?.Dispose();
            await NextFrame();

            int total = metaList.Count;
            Logger.Info($"[TextureRefine] GPU compute bake: {total} keyframes" +
                (keyframeRelocation != Matrix4x4.identity ? " (relocated)" : ""));

            int atlasW = mesh.AtlasWidth;
            int atlasH = mesh.AtlasHeight;
            int texelCount = atlasW * atlasH;
            int origTriCount = mesh.OrigIndices.Length / 3;
            int outTriCount = mesh.Indices.Length / 3;

            // Find kernels
            int kClear = compute.FindKernel("ClearDepth");
            int kDepth = compute.FindKernel("BuildDepth");
            int kBake  = compute.FindKernel("BakeAtlas");
            int kCountCharts = -1;
            try { kCountCharts = compute.FindKernel("CountChartTexels"); }
            catch { /* shader variant without coverage count */ }
            int kClearU = -1;
            int kCopyU = -1;
            try { kClearU = compute.FindKernel("ClearBuffer"); }
            catch { /* shader variant without the GPU zero kernel */ }
            try { kCopyU = compute.FindKernel("CopyUint"); }
            catch { /* shader variant without the GPU copy kernel */ }

            var uploadScope = _profile?.Stage("meshUpload");
            ReportStatus("Uploading mesh to GPU...");
            Vector2[] rawUV2 = null;
            var rawUVs = mesh.RawUVs;
            int vertCount = mesh.Positions.Length;
            await Task.Run(() =>
            {
                rawUV2 = new Vector2[vertCount];
                for (int i = 0; i < rawUV2.Length; i++)
                    rawUV2[i] = new Vector2(rawUVs[i * 2], rawUVs[i * 2 + 1]);
            });

            // ── Create persistent GPU buffers (mesh data + atlas) ──
            var origPosBuf = new ComputeBuffer(mesh.OrigPositions.Length, 12);
            origPosBuf.SetData(mesh.OrigPositions);
            var origIdxBuf = new ComputeBuffer(mesh.OrigIndices.Length, 4);
            origIdxBuf.SetData(mesh.OrigIndices);

            var outPosBuf = new ComputeBuffer(mesh.Positions.Length, 12);
            outPosBuf.SetData(mesh.Positions);
            var outNormBuf = new ComputeBuffer(mesh.Normals.Length, 12);
            outNormBuf.SetData(mesh.Normals);
            var outIdxBuf = new ComputeBuffer(mesh.Indices.Length, 4);
            outIdxBuf.SetData(mesh.Indices);

            var rawUVBuf = new ComputeBuffer(rawUV2.Length, 8);
            rawUVBuf.SetData(rawUV2);

            var scoreBuf = new ComputeBuffer(texelCount, 4);
            await ZeroUintBuffer(compute, kClearU, scoreBuf, texelCount);
            var atlasBuf = new ComputeBuffer(texelCount, 4);
            await ZeroUintBuffer(compute, kClearU, atlasBuf, texelCount);

            // ── Chart preference: per-triangle chart ids and a charts×keyframes
            // score table. Off when the unwrap came back from disk without charts.
            int chartCount = mesh.ChartIndices != null && mesh.ChartCount > 0 && chartBestViewBoost > 1.001f
                ? mesh.ChartCount : 0;
            // Always bound, 1 element when off: an unbound StructuredBuffer is
            // undefined on Vulkan even behind a branch that never reads it.
            var triChartBuf = new ComputeBuffer(Mathf.Max(1, chartCount > 0 ? outTriCount : 1), 4);
            var chartScoreBuf = new ComputeBuffer(Mathf.Max(1, chartCount * total), 4);
            var chartCoverBuf = new ComputeBuffer(Mathf.Max(1, chartCount * total), 4);
            var chartSizeBuf = new ComputeBuffer(Mathf.Max(1, chartCount), 4);
            var chartBestBuf = new ComputeBuffer(Mathf.Max(1, chartCount), 4);
            if (chartCount > 0)
            {
                int[] triChart = null;
                var chartIdx = mesh.ChartIndices;
                var idx = mesh.Indices;
                await Task.Run(() =>
                {
                    triChart = new int[outTriCount];
                    for (int t = 0; t < outTriCount; t++)
                    {
                        int v = idx[t * 3];
                        triChart[t] = v < chartIdx.Length ? chartIdx[v] : -1;
                    }
                });
                triChartBuf.SetData(triChart);
                await ZeroUintBuffer(compute, kClearU, chartScoreBuf, chartCount * total);
                await ZeroUintBuffer(compute, kClearU, chartCoverBuf, chartCount * total);
                await ZeroUintBuffer(compute, kClearU, chartSizeBuf, chartCount);
            }
            else
            {
                triChartBuf.SetData(new[] { -1 });
                chartBestBuf.SetData(new[] { -1 });
            }
            compute.SetBuffer(kBake, "_TriChart", triChartBuf);
            compute.SetBuffer(kBake, "_ChartScore", chartScoreBuf);
            compute.SetBuffer(kBake, "_ChartCover", chartCoverBuf);
            compute.SetInt("_ChartCount", chartCount);
            compute.SetInt("_KfCount", total);
            compute.SetFloat("_ChartBoost", chartBestViewBoost);
            BindViewSelection(compute);

            // Bind static buffers to kernels
            compute.SetBuffer(kDepth, "_OrigPos", origPosBuf);
            compute.SetBuffer(kDepth, "_OrigIdx", origIdxBuf);
            compute.SetBuffer(kBake, "_OutPos", outPosBuf);
            compute.SetBuffer(kBake, "_OutNorm", outNormBuf);
            compute.SetBuffer(kBake, "_OutIdx", outIdxBuf);
            compute.SetBuffer(kBake, "_RawUV", rawUVBuf);
            compute.SetBuffer(kBake, "_ScoreBuf", scoreBuf);
            compute.SetBuffer(kBake, "_AtlasBuf", atlasBuf);
            compute.SetInt("_AtlasW", atlasW);
            compute.SetInt("_AtlasH", atlasH);
            compute.SetInt("_OrigTriCount", origTriCount);
            compute.SetInt("_OutTriCount", outTriCount);

            // Texel → triangle map, once. The per-keyframe kernels are one
            // thread per texel from here on.
            var texelTriBuf = new ComputeBuffer(texelCount, 4);
            await ZeroUintBuffer(compute, kClearU, texelTriBuf, texelCount);
            {
                int kMap = compute.FindKernel("BuildTexelMap");
                compute.SetBuffer(kMap, "_OutIdx", outIdxBuf);
                compute.SetBuffer(kMap, "_RawUV", rawUVBuf);
                compute.SetBuffer(kMap, "_TexelTriRW", texelTriBuf);
                compute.Dispatch(kMap, (outTriCount + 63) / 64, 1, 1);
                await NextFrame();
            }
            compute.SetBuffer(kBake, "_TexelTri", texelTriBuf);
            int texelGroups = (texelCount + 255) / 256;

            if (chartCount > 0 && kCountCharts >= 0)
            {
                compute.SetBuffer(kCountCharts, "_TexelTri", texelTriBuf);
                compute.SetBuffer(kCountCharts, "_TriChart", triChartBuf);
                compute.SetBuffer(kCountCharts, "_ChartSize", chartSizeBuf);
                compute.Dispatch(kCountCharts, texelGroups, 1, 1);
            }

            // Seam pairs are CPU work over the unwrap only — build them on a
            // worker while the GPU bakes and consume them after the blend.
            Task<SeamPair[]> seamPairsTask = null;
            if (enableSeamBlending)
            {
                var sPos = mesh.Positions; var sUV = mesh.RawUVs; var sIdx = mesh.Indices;
                seamPairsTask = Task.Run(() => BuildSeamPairs(sPos, sUV, sIdx, atlasW, atlasH));
            }

            uploadScope?.Dispose();

            var pass1Scope = _profile?.Stage("pass1");
            ReportStatus("Baking textures (GPU compute)...");
            int bakeCount = await ProcessKeyframesGpuAsync(
                metaList, compute, kClear, kDepth, origTriCount, "Baking (GPU)...",
                (tex, kf, depth) =>
                {
                    compute.SetBuffer(kBake, "_DepthBuf", depth);
                    compute.SetTexture(kBake, "_KfTex", tex);
                    compute.Dispatch(kBake, texelGroups, 1, 1);
                });

            Logger.Info($"[TextureRefine] GPU baked {bakeCount} keyframes total (pass 1)");
            pass1Scope?.Dispose();

            if (chartCount > 0)
            {
                int kArgmax = compute.FindKernel("ChartArgmax");
                compute.SetBuffer(kArgmax, "_ChartScore", chartScoreBuf);
                compute.SetBuffer(kArgmax, "_ChartCover", chartCoverBuf);
                compute.SetBuffer(kArgmax, "_ChartSize", chartSizeBuf);
                compute.SetBuffer(kArgmax, "_ChartBestRW", chartBestBuf);
                compute.Dispatch(kArgmax, (chartCount + 63) / 64, 1, 1);
                await NextFrame();
                Logger.Info($"[TextureRefine] Chart preference: {chartCount} charts, boost ×{chartBestViewBoost:F1}, minCover={chartMinCover:F2}");
            }

            // ── Registration: align each keyframe to the pass-1 atlas before it
            // is blended. Needs the blend pass to have any effect on colour.
            RegistrationScratch reg = null;
            ComputeBuffer atlasSnapshot = null;
            System.Func<Texture2D, Keyframe, Task<Keyframe>> refineHook = null;
            if (refineKeyframePoses && multiViewBlend)
            {
                int kRender = -1, kDown = -1, kMatch = -1, kGain = -1;
                try
                {
                    kRender = compute.FindKernel("RenderAtlasView");
                    kDown = compute.FindKernel("KfLumDownsample");
                    kMatch = compute.FindKernel("MatchShift");
                }
                catch { /* shader variant without registration kernels */ }
                try { kGain = compute.FindKernel("ViewGainReduce"); }
                catch { /* shader variant without exposure equalisation */ }

                if (kRender >= 0 && kDown >= 0 && kMatch >= 0)
                {
                    atlasSnapshot = new ComputeBuffer(texelCount, 4);
                    await CopyUintBuffer(compute, kCopyU, atlasBuf, atlasSnapshot, texelCount);
                    reg = new RegistrationScratch(4, registrationSearchRadius);
                    compute.SetBuffer(kRender, "_OutPos", outPosBuf);
                    compute.SetBuffer(kRender, "_OutIdx", outIdxBuf);
                    compute.SetBuffer(kRender, "_RawUV", rawUVBuf);
                    var regScratch = reg;
                    refineHook = (tex, kf) => RefineKeyframeAsync(
                        compute, regScratch, kClear, kClearU, kDepth, kRender, kDown, kMatch, kGain,
                        origTriCount, outTriCount, atlasSnapshot, tex, kf);
                }
            }

            // ── Pass 2: Multi-view blend accumulation (optional) ──
            // Re-iterates keyframes, accumulating score-weighted colors from all
            // qualifying views into fixed-point buffers, then resolves to final atlas.
            if (multiViewBlend)
            {
                int kAccum = compute.FindKernel("BlendAccum");
                int kResolve = compute.FindKernel("ResolveBlend");

                var accumR = new ComputeBuffer(texelCount, 4);
                var accumG = new ComputeBuffer(texelCount, 4);
                var accumB = new ComputeBuffer(texelCount, 4);
                var accumW = new ComputeBuffer(texelCount, 4);
                var accumN = new ComputeBuffer(texelCount, 4);
                await ZeroUintBuffer(compute, kClearU, accumR, texelCount);
                await ZeroUintBuffer(compute, kClearU, accumG, texelCount);
                await ZeroUintBuffer(compute, kClearU, accumB, texelCount);
                await ZeroUintBuffer(compute, kClearU, accumW, texelCount);
                await ZeroUintBuffer(compute, kClearU, accumN, texelCount);

                compute.SetBuffer(kAccum, "_OutPos", outPosBuf);
                compute.SetBuffer(kAccum, "_OutNorm", outNormBuf);
                compute.SetBuffer(kAccum, "_OutIdx", outIdxBuf);
                compute.SetBuffer(kAccum, "_RawUV", rawUVBuf);
                compute.SetBuffer(kAccum, "_AccumR", accumR);
                compute.SetBuffer(kAccum, "_AccumG", accumG);
                compute.SetBuffer(kAccum, "_AccumB", accumB);
                compute.SetBuffer(kAccum, "_AccumW", accumW);
                compute.SetBuffer(kAccum, "_AccumN", accumN);
                compute.SetBuffer(kAccum, "_BestScore", scoreBuf);
                compute.SetFloat("_BlendMinFraction", blendMinFraction);
                compute.SetInt("_MaxViews", Mathf.Max(1, maxViewsPerTexel));
                compute.SetBuffer(kAccum, "_TriChart", triChartBuf);
                compute.SetBuffer(kAccum, "_ChartBest", chartBestBuf);
                compute.SetBuffer(kAccum, "_TexelTri", texelTriBuf);

                var pass2Scope = _profile?.Stage(refineHook != null ? "pass2+registration" : "pass2");
                ReportStatus(refineHook != null ? "Registering + blending (pass 2)..." : "Multi-view blending (pass 2)...");
                int[] blendOrder = null;
                if (chartCount > 0)
                {
                    byte[] raw = await ReadbackComputeBufferAsync(chartScoreBuf, chartCount * total);
                    var table = new uint[chartCount * total];
                    Buffer.BlockCopy(raw, 0, table, 0, raw.Length);
                    blendOrder = BlendOrderFromChartScores(table, chartCount, total);
                }
                int blendCount = await ProcessKeyframesGpuAsync(
                    metaList, compute, kClear, kDepth, origTriCount, "Multi-view blend...",
                    (tex, kf, depth) =>
                    {
                        compute.SetBuffer(kAccum, "_DepthBuf", depth);
                        compute.SetTexture(kAccum, "_KfTex", tex);
                        compute.SetVector("_KfGain", kf.GainOrOne);
                        compute.Dispatch(kAccum, texelGroups, 1, 1);
                    },
                    refineHook, blendOrder);

                Logger.Info($"[TextureRefine] Blend pass 2 complete: {blendCount} keyframes");
                if (reg != null)
                {
                    Logger.Info($"[TextureRefine] Registration: {reg.Refined} refined, {reg.Rejected} rejected, " +
                                $"mean shift {(reg.Refined > 0 ? reg.SumShiftPx / reg.Refined : 0f):F1} px, " +
                                $"max {reg.MaxShiftPx:F1} px (radius ±{reg.Radius * reg.Scale} px, minNcc {registrationMinNcc:F2}); " +
                                $"exposure gain on {reg.Gained} keyframes, luma {reg.MinGain:F2}..{reg.MaxGain:F2}; " +
                                $"mean sharpness weight refined {(reg.Refined > 0 ? reg.RefinedWeight / reg.Refined : 0f):F2}, " +
                                $"rejected {(reg.Rejected > 0 ? reg.RejectedWeight / reg.Rejected : 0f):F2}");
                }

                // Resolve: divide accumulated colors by weights → final atlas
                compute.SetBuffer(kResolve, "_AccumRIn", accumR);
                compute.SetBuffer(kResolve, "_AccumGIn", accumG);
                compute.SetBuffer(kResolve, "_AccumBIn", accumB);
                compute.SetBuffer(kResolve, "_AccumWIn", accumW);
                compute.SetBuffer(kResolve, "_AtlasBuf", atlasBuf);
                compute.Dispatch(kResolve, (texelCount + 255) / 256, 1, 1);

                accumR.Release(); accumG.Release();
                accumB.Release(); accumW.Release(); accumN.Release();
                Logger.Info($"[TextureRefine] Multi-view blend resolved (minFraction={blendMinFraction:F2}, maxViews={maxViewsPerTexel})");
                pass2Scope?.Dispose();
            }

            // ── Sharpening pass (GPU unsharp mask) ──
            if (sharpenStrength > 0.01f)
            {
                using var _ = _profile?.Stage("sharpen") ?? default;
                ReportStatus("Sharpening...");
                int kSharpen = compute.FindKernel("SharpenAtlas");
                var sharpenSrcBuf = new ComputeBuffer(texelCount, 4);
                await CopyUintBuffer(compute, kCopyU, atlasBuf, sharpenSrcBuf, texelCount);

                compute.SetFloat("_SharpenStrength", sharpenStrength);
                compute.SetInt("_SharpenRadius", sharpenRadius);
                compute.SetInt("_AtlasW", atlasW);
                compute.SetInt("_AtlasH", atlasH);
                compute.SetBuffer(kSharpen, "_AtlasBufSrc", sharpenSrcBuf);
                compute.SetBuffer(kSharpen, "_AtlasBuf", atlasBuf);

                int groupsX = (atlasW + 7) / 8;
                int groupsY = (atlasH + 7) / 8;
                compute.Dispatch(kSharpen, groupsX, groupsY, 1);

                sharpenSrcBuf.Release();
                Logger.Info($"[TextureRefine] Sharpening complete (strength={sharpenStrength}, radius={sharpenRadius})");
            }

            // ── Seam levelling pass (GPU, spread over frames) ──
            if (seamPairsTask != null)
            {
                SeamPair[] pairs = null;
                try { pairs = await seamPairsTask; }
                catch (Exception e) { Logger.Warning($"[TextureRefine] Seam pairs failed: {e.Message}"); }

                int kDelta = -1, kDiffuse = -1, kApply = -1;
                try
                {
                    kDelta = compute.FindKernel("SeamDelta");
                    kDiffuse = compute.FindKernel("SeamDiffuse");
                    kApply = compute.FindKernel("SeamApply");
                }
                catch { /* shader variant without seam kernels */ }

                if (pairs != null && pairs.Length > 0 && kDelta >= 0 && kDiffuse >= 0 && kApply >= 0)
                {
                    using var _ = _profile?.Stage("seamLevel") ?? default;
                    ReportStatus("Levelling seams...");
                    var seamSrcBuf = new ComputeBuffer(texelCount, 4);
                    await CopyUintBuffer(compute, kCopyU, atlasBuf, seamSrcBuf, texelCount);
                    var pairBuf = new ComputeBuffer(pairs.Length, 8);
                    pairBuf.SetData(pairs);
                    var corrA = new ComputeBuffer(texelCount, 4);
                    var corrB = new ComputeBuffer(texelCount, 4);
                    await ZeroUintBuffer(compute, kClearU, corrA, texelCount);

                    compute.SetInt("_AtlasW", atlasW);
                    compute.SetInt("_AtlasH", atlasH);
                    compute.SetInt("_SeamPairCount", pairs.Length);
                    compute.SetBuffer(kDelta, "_SeamPairs", pairBuf);
                    compute.SetBuffer(kDelta, "_AtlasBufSrc", seamSrcBuf);
                    compute.SetBuffer(kDelta, "_CorrOut", corrA);
                    compute.Dispatch(kDelta, (pairs.Length + 255) / 256, 1, 1);
                    await NextFrame();

                    int groupsX = (atlasW + 7) / 8;
                    int groupsY = (atlasH + 7) / 8;
                    compute.SetBuffer(kDiffuse, "_AtlasBufSrc", seamSrcBuf);
                    var src = corrA; var dst = corrB;
                    int iterations = Mathf.Max(1, seamLevelIterations);
                    for (int it = 0; it < iterations; it++)
                    {
                        compute.SetBuffer(kDiffuse, "_CorrIn", src);
                        compute.SetBuffer(kDiffuse, "_CorrOut", dst);
                        compute.Dispatch(kDiffuse, groupsX, groupsY, 1);
                        (src, dst) = (dst, src);
                        // One step a frame: an 8-tap pass over the whole atlas.
                        await NextFrame();
                    }

                    compute.SetBuffer(kApply, "_AtlasBufSrc", seamSrcBuf);
                    compute.SetBuffer(kApply, "_CorrIn", src);
                    compute.SetBuffer(kApply, "_AtlasBuf", atlasBuf);
                    compute.Dispatch(kApply, (texelCount + 255) / 256, 1, 1);
                    await NextFrame();

                    seamSrcBuf.Release(); pairBuf.Release(); corrA.Release(); corrB.Release();
                    Logger.Info($"[TextureRefine] Seam levelling: {pairs.Length} seam samples, {iterations} diffusion steps");
                }
                else if (pairs != null)
                {
                    Logger.Info("[TextureRefine] Seam levelling skipped: no seam samples");
                }
            }

            // ── Sobel normal map pass (GPU) ──
            ComputeBuffer normalBuf = null;
            if (normalStrength > 0.01f)
            {
                using var _ = _profile?.Stage("sobel") ?? default;
                ReportStatus("Generating normal map...");
                int kSobel = compute.FindKernel("SobelNormalMap");
                normalBuf = new ComputeBuffer(texelCount, 4);

                var sobelSrcBuf = new ComputeBuffer(texelCount, 4);
                await CopyUintBuffer(compute, kCopyU, atlasBuf, sobelSrcBuf, texelCount);

                compute.SetFloat("_NormalStrength", normalStrength);
                compute.SetInt("_AtlasW", atlasW);
                compute.SetInt("_AtlasH", atlasH);
                compute.SetBuffer(kSobel, "_AtlasBufSrc", sobelSrcBuf);
                compute.SetBuffer(kSobel, "_NormalBuf", normalBuf);

                int groupsX = (atlasW + 7) / 8;
                int groupsY = (atlasH + 7) / 8;
                compute.Dispatch(kSobel, groupsX, groupsY, 1);

                sobelSrcBuf.Release();
                Logger.Info($"[TextureRefine] Sobel normal map generated (strength={normalStrength})");
            }

            // Readback atlas buffer
            var readbackScope = _profile?.Stage("atlasReadback");
            ReportStatus("Reading back atlas...");
            byte[] atlasPixels = await ReadbackComputeBufferAsync(atlasBuf, texelCount);

            byte[] normalPixels = null;
            if (normalBuf != null)
            {
                normalPixels = await ReadbackComputeBufferAsync(normalBuf, texelCount);
                normalBuf.Release();
            }

            readbackScope?.Dispose();
            _profile?.Frame();

            // Log fill stats (worker: 4-5 M texels is a frame of main-thread time)
            {
                int filled = await Task.Run(() =>
                {
                    int n = 0;
                    for (int i = 0; i < texelCount; i++)
                        if (atlasPixels[i * 4 + 3] != 0) n++;
                    return n;
                });
                Logger.Info($"[TextureRefine] GPU bake pre-dilation: {filled}/{texelCount} texels filled " +
                    $"({100f * filled / texelCount:F1}%)");
            }

            // Post-process on background thread
            if (!skipDenoise)
            {
                using var _ = _profile?.Stage("denoise") ?? default;
                ReportStatus("Denoising...");
                await Task.Run(() => DenoiseAtlas(atlasPixels, atlasW, atlasH));
            }

            var dilateScope = _profile?.Stage("dilate");
            ReportStatus("Filling gaps...");
            await Task.Run(() =>
            {
                DilateAtlas(atlasPixels, atlasW, atlasH, 8);
                if (normalPixels != null)
                    DilateAtlas(normalPixels, atlasW, atlasH, 8);
            });
            dilateScope?.Dispose();

            // Cleanup
            origPosBuf.Release(); origIdxBuf.Release();
            outPosBuf.Release(); outNormBuf.Release(); outIdxBuf.Release();
            rawUVBuf.Release(); scoreBuf.Release(); atlasBuf.Release();
            triChartBuf.Release(); chartScoreBuf.Release(); chartCoverBuf.Release();
            chartSizeBuf.Release(); chartBestBuf.Release();
            texelTriBuf.Release();
            atlasSnapshot?.Release();
            reg?.Dispose();

            ReportStatus("Done");
            Logger.Info($"[TextureRefine] GPU compute bake complete: {atlasW}x{atlasH} atlas");
            if (_profile != null)
            {
                Logger.Info(_profile.Report($"bake {total} keyframes → {atlasW}x{atlasH}"));
                _profile = null;
            }

            return (atlasPixels, normalPixels);
        }

        // ═══════════════════════════════════════════════════════════════
        //  GPU COMPUTE HELPERS
        // ═══════════════════════════════════════════════════════════════

        static Texture2D MakeKfTexture()
        {
            var tex = new Texture2D(8, 8, TextureFormat.RGBA32, false, true)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
            };
            return tex;
        }

        async Task ZeroUintBuffer(ComputeShader compute, int kClear, ComputeBuffer buf, int count)
        {
            if (kClear >= 0)
            {
                compute.SetInt("_ClearCount", count);
                compute.SetBuffer(kClear, "_ClearBuf", buf);
                compute.Dispatch(kClear, (count + 255) / 256, 1, 1);
                await Task.Yield();
                return;
            }
            var zeros = await Task.Run(() => new uint[count]);
            buf.SetData(zeros);
            await Task.Yield();
        }

        async Task CopyUintBuffer(
            ComputeShader compute, int kCopy, ComputeBuffer src, ComputeBuffer dst, int count)
        {
            if (kCopy >= 0)
            {
                compute.SetInt("_ClearCount", count);
                compute.SetBuffer(kCopy, "_CopySrc", src);
                compute.SetBuffer(kCopy, "_CopyDst", dst);
                compute.Dispatch(kCopy, (count + 255) / 256, 1, 1);
                await Task.Yield();
                return;
            }
            var tmp = new uint[count];
            src.GetData(tmp);
            dst.SetData(tmp);
            await Task.Yield();
        }

        /// <summary>
        /// Pass-2 walk order: keyframes with the highest summed hero-chart
        /// score first, so <c>maxViewsPerTexel</c> keeps the top-K, not capture order.
        /// </summary>
        static int[] BlendOrderFromChartScores(uint[] table, int chartCount, int kfCount)
        {
            var q = new float[kfCount];
            for (int c = 0; c < chartCount; c++)
            {
                int row = c * kfCount;
                for (int k = 0; k < kfCount; k++)
                    q[k] += table[row + k];
            }
            var order = new int[kfCount];
            for (int i = 0; i < kfCount; i++) order[i] = i;
            System.Array.Sort(order, (a, b) => q[b].CompareTo(q[a]));
            return order;
        }

        /// <summary>
        /// One keyframe per compositor frame, pipelined: keyframe N+1 is read
        /// and JPEG-decoded on a worker while N's GPU work is in flight, so the
        /// main thread only uploads pixels, binds the camera and issues
        /// clear → depth → shade in one submission, then yields once. Two
        /// Texture2D slots alternate; Unity orders the upload for N+1 behind
        /// N's dispatch in the command stream, so no fence is needed.
        /// </summary>
        /// <param name="refine">Optional: runs after the keyframe is uploaded and
        /// before its depth pass, may return a corrected keyframe (pose, gain).
        /// The corrected keyframe is written back to <paramref name="metaList"/>.</param>
        /// <param name="processOrder">Keyframe indices to walk. Pass 2 supplies
        /// best-hero-first so <c>maxViewsPerTexel</c> keeps the top-K, not capture order.</param>
        async Task<int> ProcessKeyframesGpuAsync(
            System.Collections.Generic.List<Keyframe> metaList,
            ComputeShader compute,
            int kClear, int kDepth,
            int origTriCount,
            string statusPrefix,
            System.Action<Texture2D, Keyframe, ComputeBuffer> shade,
            System.Func<Texture2D, Keyframe, Task<Keyframe>> refine = null,
            int[] processOrder = null)
        {
            int First(int from)
            {
                if (processOrder != null)
                {
                    for (int s = from; s < processOrder.Length; s++)
                    {
                        int i = processOrder[s];
                        if (i >= 0 && i < metaList.Count && !string.IsNullOrEmpty(metaList[i].JpgPath))
                            return s;
                    }
                    return -1;
                }
                for (int i = from; i < metaList.Count; i++)
                    if (!string.IsNullOrEmpty(metaList[i].JpgPath)) return i;
                return -1;
            }

            int IndexAt(int cursor) => processOrder != null ? processOrder[cursor] : cursor;

            var slots = new[] { MakeKfTexture(), MakeKfTexture() };
            ComputeBuffer depthBuf = null;
            int baked = 0;

            // A JPEG decode is ~37 ms on a Quest worker core and a keyframe's
            // GPU work is one frame, so a single decode in flight left the
            // bake waiting on the worker two frames out of three. Keep a few
            // in flight; each holds one decoded frame (~5 MB) until consumed.
            var pending = new System.Collections.Generic.Queue<(int index, Task<KeyframeImageDecoder.Decoded> task)>();
            int nextToQueue = First(0);
            void FillPrefetch()
            {
                while (nextToQueue >= 0 && pending.Count < DecodePrefetchDepth)
                {
                    int i = IndexAt(nextToQueue);
                    pending.Enqueue((i, KeyframeImageDecoder.ReadAndDecodeAsync(metaList[i].JpgPath)));
                    nextToQueue = First(nextToQueue + 1);
                }
            }
            FillPrefetch();

            try
            {
                int slot = 0;
                var sw = new System.Diagnostics.Stopwatch();
                while (pending.Count > 0)
                {
                    var (i, task) = pending.Dequeue();
                    FillPrefetch();

                    KeyframeImageDecoder.Decoded img = null;
                    sw.Restart();
                    bool waited = !task.IsCompleted;
                    try { img = await task; }
                    catch (Exception e) { Logger.Warning($"[TextureRefine] Keyframe {i} unreadable: {e.Message}"); }
                    if (waited) _profile?.Frame();
                    double decodeWaitMs = sw.Elapsed.TotalMilliseconds;

                    var tex = slots[slot];
                    sw.Restart();
                    bool ok = img != null && img.ApplyTo(tex);
                    double uploadMs = sw.Elapsed.TotalMilliseconds;
                    double readMs = img?.ReadMs ?? 0, decodeMs = img?.DecodeMs ?? 0;
                    bool fallback = img?.IsFallback ?? false;
                    img?.Dispose();
                    if (!ok) continue;

                    var kf = metaList[i];
                    kf.Width = tex.width;
                    kf.Height = tex.height;
                    sw.Restart();
                    if (refine != null)
                    {
                        kf = await refine(tex, kf);
                        _profile?.Frame();
                    }
                    double refineMs = sw.Elapsed.TotalMilliseconds;
                    metaList[i] = kf;
                    sw.Restart();

                    // Occlusion depth at a fraction of the photo (dense mesh,
                    // atomics per covered pixel — the expensive raster), then
                    // the shade at full photo resolution reading it scaled.
                    float depthScale = 1f / Mathf.Max(1, occlusionDepthDivisor);
                    int depthW = Mathf.Max(1, Mathf.RoundToInt(kf.Width * depthScale));
                    int depthH = Mathf.Max(1, Mathf.RoundToInt(kf.Height * depthScale));
                    int depthPixels = depthW * depthH;
                    if (depthBuf == null || depthBuf.count != depthPixels)
                    {
                        depthBuf?.Release();
                        depthBuf = new ComputeBuffer(depthPixels, 4);
                    }

                    compute.SetInt("_KfIndex", i);
                    BindBodyCapsules(compute, kf);
                    BindKeyframeCamera(compute, kf, depthScale);
                    compute.SetBuffer(kClear, "_DepthBuf", depthBuf);
                    compute.SetBuffer(kDepth, "_DepthBuf", depthBuf);
                    compute.Dispatch(kClear, (depthPixels + 255) / 256, 1, 1);
                    compute.Dispatch(kDepth, (origTriCount + 63) / 64, 1, 1);

                    BindKeyframeCamera(compute, kf, 1f);
                    compute.SetInt("_DepthW", depthW);
                    compute.SetInt("_DepthH", depthH);
                    compute.SetFloat("_DepthScale", (float)depthW / kf.Width);
                    shade(tex, kf, depthBuf);
                    _profile?.Keyframe(decodeWaitMs, uploadMs, refineMs, sw.Elapsed.TotalMilliseconds,
                        readMs, decodeMs, fallback);

                    baked++;
                    if (baked % 20 == 0 || baked < 3)
                    {
                        ReportStatus($"{statusPrefix} {baked}/{metaList.Count}");
                        Logger.Info($"[TextureRefine] {statusPrefix} {baked}/{metaList.Count}");
                    }

                    slot = 1 - slot;
                    await NextFrame();
                }
            }
            finally
            {
                while (pending.Count > 0)
                {
                    try { (await pending.Dequeue().task)?.Dispose(); } catch { }
                }
                foreach (var t in slots)
                {
                    if (t == null) continue;
                    if (Application.isPlaying) UnityEngine.Object.Destroy(t);
                    else UnityEngine.Object.DestroyImmediate(t);
                }
                depthBuf?.Release();
            }

            return baked;
        }

        // ═══════════════════════════════════════════════════════════════
        //  SEAM PAIRS
        // ═══════════════════════════════════════════════════════════════

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        internal struct SeamPair
        {
            public uint A, B;
        }

        /// <summary>
        /// Points along every UV seam as pairs of atlas texels that sit on the
        /// same 3D point. xatlas splits a vertex at a seam, so the two copies
        /// share an exact position; an edge whose endpoints share positions
        /// but not output vertices is drawn on two charts. Sampled one texel
        /// apart along the longer side, nudged half a texel into each
        /// triangle so the sample lands on a shaded texel, not the border.
        /// Pure CPU over the unwrap — runs on a worker.
        /// </summary>
        internal static SeamPair[] BuildSeamPairs(Vector3[] pos, float[] rawUV, int[] idx, int atlasW, int atlasH)
        {
            if (pos == null || rawUV == null || idx == null || pos.Length == 0) return System.Array.Empty<SeamPair>();

            var canonOf = new System.Collections.Generic.Dictionary<Vector3, int>(pos.Length);
            var canon = new int[pos.Length];
            for (int i = 0; i < pos.Length; i++)
            {
                if (!canonOf.TryGetValue(pos[i], out int c))
                {
                    c = canonOf.Count;
                    canonOf.Add(pos[i], c);
                }
                canon[i] = c;
            }

            // (a, b, opposite) per directed edge, keyed by canonical endpoints.
            var edges = new System.Collections.Generic.Dictionary<long, System.Collections.Generic.List<(int a, int b, int o)>>();
            int triCount = idx.Length / 3;
            for (int t = 0; t < triCount; t++)
            {
                for (int e = 0; e < 3; e++)
                {
                    int va = idx[t * 3 + e], vb = idx[t * 3 + (e + 1) % 3], vo = idx[t * 3 + (e + 2) % 3];
                    int ca = canon[va], cb = canon[vb];
                    if (ca == cb) continue;
                    long key = ca < cb ? ((long)ca << 32) | (uint)cb : ((long)cb << 32) | (uint)ca;
                    var entry = ca < cb ? (va, vb, vo) : (vb, va, vo);
                    if (!edges.TryGetValue(key, out var list))
                    {
                        list = new System.Collections.Generic.List<(int, int, int)>(2);
                        edges.Add(key, list);
                    }
                    list.Add(entry);
                }
            }

            Vector2 UV(int v) => new Vector2(rawUV[v * 2], rawUV[v * 2 + 1]);
            uint Texel(Vector2 p)
            {
                int x = Mathf.Clamp((int)p.x, 0, atlasW - 1);
                int y = Mathf.Clamp((int)p.y, 0, atlasH - 1);
                return (uint)(y * atlasW + x);
            }

            var pairs = new System.Collections.Generic.List<SeamPair>(edges.Count);
            foreach (var kv in edges)
            {
                var list = kv.Value;
                if (list.Count < 2) continue;
                var e0 = list[0];
                (int a, int b, int o) e1 = default;
                bool found = false;
                for (int k = 1; k < list.Count && !found; k++)
                {
                    if (list[k].a != e0.a || list[k].b != e0.b) { e1 = list[k]; found = true; }
                }
                if (!found) continue;   // same output verts on both sides: an interior edge

                Vector2 a0 = UV(e0.a), a1 = UV(e0.b), b0 = UV(e1.a), b1 = UV(e1.b);
                Vector2 oa = UV(e0.o), ob = UV(e1.o);
                float len = Mathf.Max((a1 - a0).magnitude, (b1 - b0).magnitude);
                int n = Mathf.Clamp(Mathf.CeilToInt(len) + 1, 2, 1024);
                for (int k = 0; k < n; k++)
                {
                    float t = (float)k / (n - 1);
                    Vector2 pa = Vector2.Lerp(a0, a1, t), pb = Vector2.Lerp(b0, b1, t);
                    // Half a texel toward the opposite vertex keeps the sample inside the chart.
                    Vector2 da = oa - pa, db = ob - pb;
                    if (da.sqrMagnitude > 1e-6f) pa += da.normalized * 0.5f;
                    if (db.sqrMagnitude > 1e-6f) pb += db.normalized * 0.5f;
                    uint ta = Texel(pa), tb = Texel(pb);
                    if (ta == tb) continue;
                    pairs.Add(new SeamPair { A = ta, B = tb });
                }
            }
            return pairs.ToArray();
        }

        /// <summary>
        /// Bind one keyframe's camera. <paramref name="scale"/> below 1 binds the
        /// same camera for an image downsampled by that factor (intrinsics and
        /// crop scale together; the pose does not).
        /// </summary>
        static void BindKeyframeCamera(ComputeShader compute, in Keyframe kf, float scale)
        {
            int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
            int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
            float cropX = (sw - kf.Width) * 0.5f;
            float cropY = (sh - kf.Height) * 0.5f;
            Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;

            compute.SetMatrix("_ViewMat", viewMat);
            compute.SetVector("_CamPos", new Vector4(kf.Position.x, kf.Position.y, kf.Position.z, 1f));
            compute.SetFloat("_Fx", kf.Fx * scale);
            compute.SetFloat("_Fy", kf.Fy * scale);
            compute.SetFloat("_Cx", kf.Cx * scale);
            compute.SetFloat("_Cy", kf.Cy * scale);
            compute.SetFloat("_CropX", cropX * scale);
            compute.SetFloat("_CropY", cropY * scale);
            compute.SetInt("_ImgW", Mathf.Max(1, Mathf.RoundToInt(kf.Width * scale)));
            compute.SetInt("_ImgH", Mathf.Max(1, Mathf.RoundToInt(kf.Height * scale)));
            compute.SetFloat("_KfWeight", kf.WeightOrOne);
        }

        // ═══════════════════════════════════════════════════════════════
        //  KEYFRAME REGISTRATION
        // ═══════════════════════════════════════════════════════════════

        /// <summary>
        /// Per-keyframe GPU state for the registration step: low-res depth,
        /// the atlas rendered into this view, the photo downsampled, and the
        /// ZNCC score per candidate shift. Allocated once per bake.
        /// </summary>
        sealed class RegistrationScratch : IDisposable
        {
            public readonly int Scale;
            public readonly int Radius;
            public ComputeBuffer DepthLow, ViewLum, KfLum, MatchOut, GainSums;
            public int LowW, LowH;
            public int Refined, Rejected, Gained;
            public float RefinedWeight, RejectedWeight;
            public float SumShiftPx, MaxShiftPx, MinGain = 1f, MaxGain = 1f;

            public RegistrationScratch(int scale, int radius)
            {
                Scale = Mathf.Max(1, scale);
                Radius = Mathf.Max(1, radius);
                int side = 2 * Radius + 1;
                MatchOut = new ComputeBuffer(side * side, 4);
                GainSums = new ComputeBuffer(7, 4);
            }

            public void EnsureImage(int imgW, int imgH)
            {
                int w = Mathf.Max(1, imgW / Scale), h = Mathf.Max(1, imgH / Scale);
                if (DepthLow != null && LowW == w && LowH == h) return;
                LowW = w; LowH = h;
                DepthLow?.Release(); ViewLum?.Release(); KfLum?.Release();
                DepthLow = new ComputeBuffer(w * h, 4);
                ViewLum = new ComputeBuffer(w * h, 4);
                KfLum = new ComputeBuffer(w * h, 4);
            }

            public void Dispose()
            {
                DepthLow?.Release(); ViewLum?.Release(); KfLum?.Release(); MatchOut?.Release(); GainSums?.Release();
                DepthLow = ViewLum = KfLum = MatchOut = GainSums = null;
            }
        }

        /// <summary>
        /// Align one decoded keyframe against the pass-1 atlas rendered from
        /// its own pose and fold the best image shift into its rotation.
        /// Everything runs at 1/<see cref="RegistrationScratch.Scale"/>
        /// resolution: one depth pass over the dense mesh, one raster of the
        /// output mesh, one downsample, one ZNCC sweep, one small readback.
        /// </summary>
        async Task<Keyframe> RefineKeyframeAsync(
            ComputeShader compute, RegistrationScratch s,
            int kClear, int kClearU, int kDepth, int kRender, int kDown, int kMatch, int kGain,
            int origTriCount, int outTriCount, ComputeBuffer atlasSnapshot,
            Texture2D tex, Keyframe kf)
        {
            s.EnsureImage(kf.Width, kf.Height);
            int lowPixels = s.LowW * s.LowH;
            float scale = 1f / s.Scale;

            BindKeyframeCamera(compute, kf, scale);
            compute.SetInt("_LowScale", s.Scale);
            compute.SetInt("_SearchR", s.Radius);
            BindBodyCapsules(compute, kf);

            compute.SetBuffer(kClear, "_DepthBuf", s.DepthLow);
            compute.Dispatch(kClear, (lowPixels + 255) / 256, 1, 1);
            compute.SetBuffer(kDepth, "_DepthBuf", s.DepthLow);
            compute.Dispatch(kDepth, (origTriCount + 63) / 64, 1, 1);

            if (kClearU >= 0)
            {
                compute.SetInt("_ClearCount", lowPixels);
                compute.SetBuffer(kClearU, "_ClearBuf", s.ViewLum);
                compute.Dispatch(kClearU, (lowPixels + 255) / 256, 1, 1);
            }

            compute.SetBuffer(kRender, "_DepthBuf", s.DepthLow);
            compute.SetBuffer(kRender, "_ViewLum", s.ViewLum);
            compute.SetBuffer(kRender, "_AtlasBufSrc", atlasSnapshot);
            compute.Dispatch(kRender, (outTriCount + 63) / 64, 1, 1);

            compute.SetTexture(kDown, "_KfTex", tex);
            compute.SetBuffer(kDown, "_KfLum", s.KfLum);
            compute.Dispatch(kDown, (s.LowW + 7) / 8, (s.LowH + 7) / 8, 1);

            compute.SetBuffer(kMatch, "_ViewLum", s.ViewLum);
            compute.SetBuffer(kMatch, "_KfLum", s.KfLum);
            compute.SetBuffer(kMatch, "_MatchOut", s.MatchOut);
            int side = 2 * s.Radius + 1;
            compute.Dispatch(kMatch, side * side, 1, 1);   // one group per candidate shift

            // Exposure gain: atlas colour this pose sees vs the photo, over
            // the covered low-res pixels. Independent of the shift result.
            bool gain = kGain >= 0 && kClearU >= 0 && equalizeExposure;
            if (gain)
            {
                compute.SetInt("_ClearCount", 7);
                compute.SetBuffer(kClearU, "_ClearBuf", s.GainSums);
                compute.Dispatch(kClearU, 1, 1, 1);
                compute.SetBuffer(kGain, "_ViewLum", s.ViewLum);
                compute.SetBuffer(kGain, "_KfLum", s.KfLum);
                compute.SetBuffer(kGain, "_GainSums", s.GainSums);
                compute.Dispatch(kGain, (lowPixels + 255) / 256, 1, 1);
            }

            // Both readbacks are requested in the same frame so they land in
            // the same frame; awaiting them in turn cost a round trip each.
            var matchTask = ReadbackComputeBufferAsync(s.MatchOut, side * side);
            var gainTask = gain ? ReadbackComputeBufferAsync(s.GainSums, 7) : null;
            byte[] raw = await matchTask;
            var scores = new float[side * side];
            Buffer.BlockCopy(raw, 0, scores, 0, raw.Length);

            if (gain)
            {
                byte[] graw = await gainTask;
                var sums = new uint[7];
                Buffer.BlockCopy(graw, 0, sums, 0, graw.Length);
                // Need a real overlap and a photo that is not black.
                if (sums[6] >= 2000 && sums[3] > 0 && sums[4] > 0 && sums[5] > 0)
                {
                    float lo = 1f / exposureGainLimit, hi = exposureGainLimit;
                    var g = new Vector3(
                        Mathf.Clamp((float)sums[0] / sums[3], lo, hi),
                        Mathf.Clamp((float)sums[1] / sums[4], lo, hi),
                        Mathf.Clamp((float)sums[2] / sums[5], lo, hi));
                    kf.Gain = g;
                    s.Gained++;
                    float lum = (g.x * 0.3f + g.y * 0.59f + g.z * 0.11f);
                    if (lum < s.MinGain) s.MinGain = lum;
                    if (lum > s.MaxGain) s.MaxGain = lum;
                }
            }

            if (!BestShift(scores, side, registrationMinNcc, out Vector2 shiftLow, out float ncc))
            {
                s.Rejected++;
                s.RejectedWeight += kf.WeightOrOne;
                return kf;
            }

            // Reject a peak sitting on the search border: the true shift is
            // beyond the window and a clamped correction is worse than none.
            if (Mathf.Abs(shiftLow.x) >= s.Radius - 0.5f || Mathf.Abs(shiftLow.y) >= s.Radius - 0.5f)
            {
                s.Rejected++;
                s.RejectedWeight += kf.WeightOrOne;
                return kf;
            }

            Vector2 shiftPx = shiftLow * s.Scale;
            kf.Rotation = ApplyImageShift(kf.Rotation, shiftPx, new Vector2(kf.Fx, kf.Fy));
            s.Refined++;
            s.RefinedWeight += kf.WeightOrOne;
            float mag = shiftPx.magnitude;
            s.SumShiftPx += mag;
            if (mag > s.MaxShiftPx) s.MaxShiftPx = mag;
            return kf;
        }

        /// <summary>
        /// Peak of the ZNCC grid with a parabolic sub-pixel fit on each axis.
        /// False when nothing reaches <paramref name="minNcc"/>.
        /// </summary>
        internal static bool BestShift(float[] scores, int side, float minNcc, out Vector2 shift, out float best)
        {
            int r = side / 2;
            int bi = -1;
            best = -3f;
            for (int i = 0; i < scores.Length; i++)
                if (scores[i] > best) { best = scores[i]; bi = i; }
            shift = Vector2.zero;
            if (bi < 0 || best < minNcc) return false;

            int bx = bi % side, by = bi / side;
            float fx = bx, fy = by;
            if (bx > 0 && bx < side - 1)
            {
                float l = scores[by * side + bx - 1], c = best, rr = scores[by * side + bx + 1];
                float d = l - 2f * c + rr;
                if (d < -1e-6f) fx += 0.5f * (l - rr) / d;
            }
            if (by > 0 && by < side - 1)
            {
                float u = scores[(by - 1) * side + bx], c = best, dn = scores[(by + 1) * side + bx];
                float d = u - 2f * c + dn;
                if (d < -1e-6f) fy += 0.5f * (u - dn) / d;
            }
            shift = new Vector2(fx - r, fy - r);
            return true;
        }

        /// <summary>
        /// Rotate a camera so that what it projected at pixel p now projects
        /// at p + <paramref name="shiftPx"/>. Camera space is x right, y up,
        /// z forward (the bake's convention); a small yaw moves projections
        /// along +x by fx·yaw, a small pitch about −x moves them along +y by
        /// fy·pitch. The correction is applied in camera space, so the world
        /// rotation becomes R · ΔR⁻¹.
        /// </summary>
        internal static Quaternion ApplyImageShift(Quaternion rotation, Vector2 shiftPx, Vector2 focal)
        {
            if (focal.x <= 1f || focal.y <= 1f) return rotation;
            float yaw = Mathf.Atan(shiftPx.x / focal.x) * Mathf.Rad2Deg;
            float pitch = -Mathf.Atan(shiftPx.y / focal.y) * Mathf.Rad2Deg;
            Quaternion delta = Quaternion.AngleAxis(pitch, Vector3.right) * Quaternion.AngleAxis(yaw, Vector3.up);
            return rotation * Quaternion.Inverse(delta);
        }

        static Task<byte[]> ReadbackComputeBufferAsync(ComputeBuffer buffer, int elementCount)
        {
            var tcs = new TaskCompletionSource<byte[]>();
            AsyncGPUReadback.Request(buffer, elementCount * 4, 0, request =>
            {
                if (request.hasError)
                {
                    Logger.Error("[TextureRefine] Compute buffer readback failed");
                    tcs.SetResult(new byte[elementCount * 4]);
                    return;
                }
                var native = request.GetData<byte>();
                byte[] managed = new byte[native.Length];
                NativeArray<byte>.Copy(native, managed, native.Length);
                tcs.SetResult(managed);
            });
            return tcs.Task;
        }

        // ═══════════════════════════════════════════════════════════════
        //  CPU BAKE FALLBACK
        // ═══════════════════════════════════════════════════════════════

        /// <summary>
        /// CPU-based atlas baking fallback.
        /// Decodes one JPEG at a time on the main thread to avoid OOM, bakes on BG thread.
        /// </summary>
        async Task<byte[]> BakeAtlasCPUAsync(
            UnwrappedMeshResult mesh, string keyframeDir, Matrix4x4 keyframeRelocation)
        {
            ReportStatus("Loading keyframe metadata...");
            var metaList = ParseKeyframeManifest(keyframeDir, keyframeRelocation);
            if (metaList.Count == 0)
                throw new InvalidOperationException("No keyframes available for baking");
            ApplySharpnessWeights(metaList);

            Logger.Info($"[TextureRefine] Found {metaList.Count} keyframes" +
                (keyframeRelocation != Matrix4x4.identity ? " (relocated)" : ""));

            ReportStatus("Baking textures...");
            Vector3[] inPos = mesh.OrigPositions;
            Vector3[] inNorm = mesh.OrigNormals;
            int[] inIdx = mesh.OrigIndices;
            Vector3[] outPos = mesh.Positions;
            Vector3[] outNorm = mesh.Normals;
            float[] rawUVs = mesh.RawUVs;
            int[] outIndices = mesh.Indices;
            int outVertCount = mesh.Positions.Length;
            int atlasW = mesh.AtlasWidth;
            int atlasH = mesh.AtlasHeight;
            int texelCount = atlasW * atlasH;
            byte[] atlasPixels = new byte[texelCount * 4];
            float[] bestScore = new float[texelCount];

            TriData[] triData = null;
            await Task.Run(() => triData = PrecomputeTriData(outPos, outNorm, rawUVs, outIndices, outVertCount));

            for (int ki = 0; ki < metaList.Count; ki++)
            {
                var kf = metaList[ki];
                if (string.IsNullOrEmpty(kf.JpgPath)) continue;

                byte[] jpgBytes;
                try { jpgBytes = File.ReadAllBytes(kf.JpgPath); }
                catch { continue; }

                var tex = new Texture2D(2, 2);
                if (!ImageConversion.LoadImage(tex, jpgBytes))
                {
                    UnityEngine.Object.Destroy(tex);
                    continue;
                }
                kf.Width = tex.width;
                kf.Height = tex.height;
                Color32[] pxColors = tex.GetPixels32();
                UnityEngine.Object.Destroy(tex);
                jpgBytes = null;
                byte[] rgba = new byte[pxColors.Length * 4];
                for (int ci = 0; ci < pxColors.Length; ci++)
                {
                    rgba[ci * 4] = pxColors[ci].r;
                    rgba[ci * 4 + 1] = pxColors[ci].g;
                    rgba[ci * 4 + 2] = pxColors[ci].b;
                    rgba[ci * 4 + 3] = 255;
                }
                kf.Pixels = rgba;

                if (ki == 0)
                {
                    Logger.Info($"[TextureRefine] KF0: pos={kf.Position}, rot={kf.Rotation}, " +
                        $"fx={kf.Fx}, fy={kf.Fy}, cx={kf.Cx}, cy={kf.Cy}, " +
                        $"imgSize={kf.Width}x{kf.Height}, pixLen={kf.Pixels.Length}");
                }

                Keyframe capturedKf = kf;
                await Task.Run(() =>
                {
                    BakeSingleKeyframe(atlasPixels, bestScore, atlasW, atlasH,
                        inPos, inNorm, inIdx, outPos, outNorm,
                        rawUVs, outIndices, outVertCount, capturedKf, triData);
                });

                if (ki % 20 == 0 || ki < 3 || ki == metaList.Count - 1)
                {
                    ReportStatus($"Baking... {ki + 1}/{metaList.Count}");
                    Logger.Info($"[TextureRefine] Baked keyframe {ki + 1}/{metaList.Count}");
                }
            }

            {
                int filled = 0;
                for (int i = 0; i < texelCount; i++)
                    if (atlasPixels[i * 4 + 3] != 0) filled++;
                Logger.Info($"[TextureRefine] Pre-dilation: {filled}/{texelCount} texels filled " +
                    $"({100f * filled / texelCount:F1}%)");
            }

            if (!skipDenoise)
            {
                ReportStatus("Denoising...");
                await Task.Run(() => DenoiseAtlas(atlasPixels, atlasW, atlasH));
            }

            ReportStatus("Filling gaps...");
            await Task.Run(() => DilateAtlas(atlasPixels, atlasW, atlasH, 8));

            ReportStatus("Done");
            Logger.Info($"[TextureRefine] Complete: {atlasW}x{atlasH} atlas");

            return atlasPixels;
        }

        static void BakeSingleKeyframe(
            byte[] atlas, float[] bestScore, int atlasW, int atlasH,
            Vector3[] inPos, Vector3[] inNorm, int[] origIndices,
            Vector3[] outPos, Vector3[] outNorm,
            float[] rawUVs, int[] indices, int outVertCount,
            Keyframe kf, TriData[] triData = null)
        {
            if (kf.Pixels == null || kf.Width == 0) return;

            float[] depthBuf = BuildDepthBuffer(inPos, origIndices, kf, kf.Width, kf.Height);

            Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;
            Vector3 camPos = kf.Position;

            int triCount = triData != null ? triData.Length : indices.Length / 3;

            for (int t = 0; t < triCount; t++)
            {
                int i0, i1, i2;
                Vector3 faceNormal, centroid;
                float u0, v0, u1, v1, u2, v2;

                if (triData != null)
                {
                    ref readonly TriData td = ref triData[t];
                    if (td.I0 < 0) continue;
                    i0 = td.I0; i1 = td.I1; i2 = td.I2;
                    faceNormal = td.FaceNormal;
                    centroid = td.Centroid;
                    u0 = td.U0; v0 = td.V0;
                    u1 = td.U1; v1 = td.V1;
                    u2 = td.U2; v2 = td.V2;
                }
                else
                {
                    i0 = indices[t * 3]; i1 = indices[t * 3 + 1]; i2 = indices[t * 3 + 2];
                    if (i0 >= outVertCount || i1 >= outVertCount || i2 >= outVertCount) continue;
                    Vector3 p = outPos[i0], q = outPos[i1], r = outPos[i2];
                    faceNormal = Vector3.Cross(q - p, r - p).normalized;
                    if (faceNormal.sqrMagnitude < 0.001f) faceNormal = outNorm[i0];
                    centroid = (p + q + r) / 3f;
                    u0 = rawUVs[i0 * 2]; v0 = rawUVs[i0 * 2 + 1];
                    u1 = rawUVs[i1 * 2]; v1 = rawUVs[i1 * 2 + 1];
                    u2 = rawUVs[i2 * 2]; v2 = rawUVs[i2 * 2 + 1];
                }

                Vector3 viewDir = (camPos - centroid).normalized;
                float dot = Vector3.Dot(faceNormal, viewDir);
                if (dot <= 0.05f) continue;

                Vector3 p0 = outPos[i0], p1 = outPos[i1], p2 = outPos[i2];
                Vector2 s0 = ProjectToScreen(p0, viewMat, kf);
                Vector2 s1 = ProjectToScreen(p1, viewMat, kf);
                Vector2 s2 = ProjectToScreen(p2, viewMat, kf);

                if (!IsInFrustum(s0, kf.Width, kf.Height) &&
                    !IsInFrustum(s1, kf.Width, kf.Height) &&
                    !IsInFrustum(s2, kf.Width, kf.Height))
                    continue;

                float dist = Vector3.Distance(camPos, centroid);
                float score = ViewScore(dot, dist, kf.Fx) * kf.WeightOrOne;

                RasterizeTriangle(atlas, bestScore, atlasW, atlasH,
                    u0, v0, u1, v1, u2, v2,
                    s0, s1, s2,
                    score, kf, depthBuf, p0, p1, p2, viewMat);
            }
        }

        // ═══════════════════════════════════════════════════════════════
        //  PROJECTION HELPERS
        // ═══════════════════════════════════════════════════════════════

        static Vector2 ProjectToScreen(Vector3 worldPos, Matrix4x4 viewMat, Keyframe kf)
        {
            Vector3 cam = viewMat.MultiplyPoint3x4(worldPos);
            if (cam.z <= 0.001f) return new Vector2(-1, -1);
            float sensorX = kf.Fx * (cam.x / cam.z) + kf.Cx;
            float sensorY = kf.Fy * (cam.y / cam.z) + kf.Cy;
            int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
            int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
            float cropX = (sw - kf.Width) * 0.5f;
            float cropY = (sh - kf.Height) * 0.5f;
            return new Vector2(sensorX - cropX, sensorY - cropY);
        }

        static bool IsInFrustum(Vector2 screen, int w, int h)
        {
            return screen.x >= -w * 0.1f && screen.x < w * 1.1f &&
                   screen.y >= -h * 0.1f && screen.y < h * 1.1f;
        }

        static float[] BuildDepthBuffer(Vector3[] positions, int[] indices,
            Keyframe kf, int w, int h)
        {
            float[] depth = new float[w * h];
            for (int i = 0; i < depth.Length; i++) depth[i] = float.MaxValue;

            Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;
            int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
            int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
            float cropX = (sw - kf.Width) * 0.5f;
            float cropY = (sh - kf.Height) * 0.5f;
            int triCount = indices.Length / 3;

            for (int t = 0; t < triCount; t++)
            {
                int i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
                if (i0 >= positions.Length || i1 >= positions.Length || i2 >= positions.Length) continue;

                Vector3 c0 = viewMat.MultiplyPoint3x4(positions[i0]);
                Vector3 c1 = viewMat.MultiplyPoint3x4(positions[i1]);
                Vector3 c2 = viewMat.MultiplyPoint3x4(positions[i2]);

                if (c0.z <= 0 && c1.z <= 0 && c2.z <= 0) continue;

                Vector2 s0 = new Vector2(kf.Fx * c0.x / Mathf.Max(c0.z, 0.001f) + kf.Cx - cropX,
                                         kf.Fy * c0.y / Mathf.Max(c0.z, 0.001f) + kf.Cy - cropY);
                Vector2 s1 = new Vector2(kf.Fx * c1.x / Mathf.Max(c1.z, 0.001f) + kf.Cx - cropX,
                                         kf.Fy * c1.y / Mathf.Max(c1.z, 0.001f) + kf.Cy - cropY);
                Vector2 s2 = new Vector2(kf.Fx * c2.x / Mathf.Max(c2.z, 0.001f) + kf.Cx - cropX,
                                         kf.Fy * c2.y / Mathf.Max(c2.z, 0.001f) + kf.Cy - cropY);

                RasterizeDepthTriangle(depth, w, h, s0, s1, s2, c0.z, c1.z, c2.z);
            }

            return depth;
        }

        static unsafe void RasterizeDepthTriangle(float[] depth, int w, int h,
            Vector2 s0, Vector2 s1, Vector2 s2,
            float z0, float z1, float z2)
        {
            int minX = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(s0.x, Mathf.Min(s1.x, s2.x))));
            int maxX = Mathf.Min(w - 1, Mathf.CeilToInt(Mathf.Max(s0.x, Mathf.Max(s1.x, s2.x))));
            int minY = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(s0.y, Mathf.Min(s1.y, s2.y))));
            int maxY = Mathf.Min(h - 1, Mathf.CeilToInt(Mathf.Max(s0.y, Mathf.Max(s1.y, s2.y))));

            float denom = (s1.y - s2.y) * (s0.x - s2.x) + (s2.x - s1.x) * (s0.y - s2.y);
            if (Mathf.Abs(denom) < 1e-8f) return;
            float invDenom = 1f / denom;

            float a0x = s1.y - s2.y, a0y = s2.x - s1.x;
            float a1x = s2.y - s0.y, a1y = s0.x - s2.x;
            float ox = -s2.x, oy = -s2.y;

            fixed (float* pDepth = depth)
            {
                for (int y = minY; y <= maxY; y++)
                {
                    float dy = y + oy;
                    float* row = pDepth + y * w;
                    for (int x = minX; x <= maxX; x++)
                    {
                        float dx = x + ox;
                        float bw0 = (a0x * dx + a0y * dy) * invDenom;
                        float bw1 = (a1x * dx + a1y * dy) * invDenom;
                        float bw2 = 1f - bw0 - bw1;

                        if (bw0 < -0.001f || bw1 < -0.001f || bw2 < -0.001f) continue;

                        float z = bw0 * z0 + bw1 * z1 + bw2 * z2;
                        if (z < row[x]) row[x] = z;
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        //  UV-SPACE TRIANGLE RASTERIZATION
        // ═══════════════════════════════════════════════════════════════

        static unsafe void RasterizeTriangle(
            byte[] atlas, float[] bestScore, int atlasW, int atlasH,
            float u0, float v0, float u1, float v1, float u2, float v2,
            Vector2 s0, Vector2 s1, Vector2 s2,
            float score, Keyframe kf, float[] depthBuf,
            Vector3 p0, Vector3 p1, Vector3 p2, Matrix4x4 viewMat)
        {
            int minX = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(u0, Mathf.Min(u1, u2))));
            int maxX = Mathf.Min(atlasW - 1, Mathf.CeilToInt(Mathf.Max(u0, Mathf.Max(u1, u2))));
            int minY = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(v0, Mathf.Min(v1, v2))));
            int maxY = Mathf.Min(atlasH - 1, Mathf.CeilToInt(Mathf.Max(v0, Mathf.Max(v1, v2))));

            float denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
            if (Mathf.Abs(denom) < 1e-8f) return;
            float invDenom = 1f / denom;

            float a0x = v1 - v2, a0y = u2 - u1;
            float a1x = v2 - v0, a1y = u0 - u2;
            float ox = -u2, oy = -v2;

            int kfW = kf.Width, kfH = kf.Height;
            int pixelLen = kf.Pixels.Length;

            fixed (byte* pAtlas = atlas, pPixels = kf.Pixels)
            fixed (float* pScore = bestScore, pDepth = depthBuf)
            {
                for (int y = minY; y <= maxY; y++)
                {
                    float dy = y + oy;
                    int atlasRowOff = y * atlasW;
                    for (int x = minX; x <= maxX; x++)
                    {
                        float dx = x + ox;
                        float bw0 = (a0x * dx + a0y * dy) * invDenom;
                        float bw1 = (a1x * dx + a1y * dy) * invDenom;
                        float bw2 = 1f - bw0 - bw1;

                        if (bw0 < -0.001f || bw1 < -0.001f || bw2 < -0.001f) continue;

                        int texelIdx = atlasRowOff + x;
                        if (score <= pScore[texelIdx]) continue;

                        float sx = bw0 * s0.x + bw1 * s1.x + bw2 * s2.x;
                        float sy = bw0 * s0.y + bw1 * s1.y + bw2 * s2.y;

                        int px = Mathf.RoundToInt(sx);
                        int screenY = Mathf.RoundToInt(sy);
                        if ((uint)px >= (uint)kfW || (uint)screenY >= (uint)kfH) continue;

                        Vector3 worldPt = bw0 * p0 + bw1 * p1 + bw2 * p2;
                        Vector3 camPt = viewMat.MultiplyPoint3x4(worldPt);
                        int depthIdx = screenY * kfW + px;
                        if (camPt.z > pDepth[depthIdx] + 0.05f) continue;

                        int pixelIdx = depthIdx * 4;
                        if (pixelIdx + 3 >= pixelLen) continue;

                        int atlasOff = texelIdx * 4;
                        pAtlas[atlasOff] = pPixels[pixelIdx];
                        pAtlas[atlasOff + 1] = pPixels[pixelIdx + 1];
                        pAtlas[atlasOff + 2] = pPixels[pixelIdx + 2];
                        pAtlas[atlasOff + 3] = 255;
                        pScore[texelIdx] = score;
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        //  DENOISE (SPECKLE REMOVAL)
        // ═══════════════════════════════════════════════════════════════

        static void DenoiseAtlas(byte[] atlas, int w, int h)
        {
            const int threshold = 60;
            byte[] clean = new byte[atlas.Length];
            Buffer.BlockCopy(atlas, 0, clean, 0, atlas.Length);

            int replaced = 0;
            for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                int idx = (y * w + x) * 4;
                if (atlas[idx + 3] == 0) continue;

                int cr = atlas[idx], cg = atlas[idx + 1], cb = atlas[idx + 2];

                int sumR = 0, sumG = 0, sumB = 0, count = 0;
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nIdx = ((y + dy) * w + (x + dx)) * 4;
                    if (atlas[nIdx + 3] == 0) continue;
                    sumR += atlas[nIdx];
                    sumG += atlas[nIdx + 1];
                    sumB += atlas[nIdx + 2];
                    count++;
                }

                if (count < 3) continue;

                int avgR = sumR / count, avgG = sumG / count, avgB = sumB / count;
                int diff = Mathf.Abs(cr - avgR) + Mathf.Abs(cg - avgG) + Mathf.Abs(cb - avgB);

                if (diff > threshold)
                {
                    clean[idx] = (byte)avgR;
                    clean[idx + 1] = (byte)avgG;
                    clean[idx + 2] = (byte)avgB;
                    replaced++;
                }
            }

            Buffer.BlockCopy(clean, 0, atlas, 0, atlas.Length);
            Logger.Info($"[TextureRefine] Denoise: replaced {replaced} outlier texels (threshold={threshold})");
        }

        // ═══════════════════════════════════════════════════════════════
        //  DILATION (GAP FILL)
        // ═══════════════════════════════════════════════════════════════

        static void DilateAtlas(byte[] atlas, int w, int h, int passes)
        {
            byte[] temp = new byte[atlas.Length];

            for (int pass = 0; pass < passes; pass++)
            {
                Buffer.BlockCopy(atlas, 0, temp, 0, atlas.Length);
                bool changed = false;

                for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int idx = (y * w + x) * 4;
                    if (atlas[idx + 3] != 0) continue; // already filled

                    int r = 0, g = 0, b = 0, count = 0;
                    for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        int nIdx = (ny * w + nx) * 4;
                        if (atlas[nIdx + 3] == 0) continue;
                        r += atlas[nIdx];
                        g += atlas[nIdx + 1];
                        b += atlas[nIdx + 2];
                        count++;
                    }

                    if (count > 0)
                    {
                        temp[idx] = (byte)(r / count);
                        temp[idx + 1] = (byte)(g / count);
                        temp[idx + 2] = (byte)(b / count);
                        temp[idx + 3] = 255;
                        changed = true;
                    }
                }

                Buffer.BlockCopy(temp, 0, atlas, 0, atlas.Length);
                if (!changed) break;
            }
        }

        /// <summary>
        /// Simplifies a baked refined mesh on a background thread, preserving UVs via
        /// meshopt_simplifyWithAttributes with UV coordinates as vertex attributes and
        /// border-locked vertices to prevent seam tearing.
        /// </summary>
        internal async Task<RefinedTextureResult> SimplifyRefinedMeshAsync(RefinedTextureResult source)
        {
            float ratio = postBakeSimplificationRatio;
            if (ratio >= 1f || simplifyBeforeUnwrap) return source;
            ratio = Mathf.Clamp(ratio, 0.05f, 1f);

            ReportStatus($"Simplifying baked mesh ({ratio:P0})...");

            RefinedTextureResult result = source;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await Task.Run(() =>
            {
                result = XAtlasWrapper.SimplifyWithUVs(
                    source.Positions, source.Normals, source.UVs,
                    source.Indices, ratio);
                result.AtlasPixels = source.AtlasPixels;
                result.NormalPixels = source.NormalPixels;
                result.AtlasWidth = source.AtlasWidth;
                result.AtlasHeight = source.AtlasHeight;
            });

            Logger.Info($"[TextureRefine] Post-bake simplify: {result.Indices.Length / 3} tris " +
                        $"(was {source.Indices.Length / 3}, ratio={ratio:F2}) in {sw.ElapsedMilliseconds} ms (worker)");
            return result;
        }

        /// <summary>
        /// Per-vertex tangents (xyz + handedness) for a normal-mapped mesh,
        /// Lengyel's method, on any thread. <c>Mesh.RecalculateTangents</c> is
        /// main-thread only and cost a frame on the refined mesh.
        /// </summary>
        internal static Vector4[] ComputeTangents(Vector3[] pos, Vector3[] nrm, Vector2[] uv, int[] idx)
        {
            int n = pos.Length;
            var tan1 = new Vector3[n];
            var tan2 = new Vector3[n];
            for (int t = 0; t + 2 < idx.Length; t += 3)
            {
                int i0 = idx[t], i1 = idx[t + 1], i2 = idx[t + 2];
                Vector3 e1 = pos[i1] - pos[i0], e2 = pos[i2] - pos[i0];
                Vector2 d1 = uv[i1] - uv[i0], d2 = uv[i2] - uv[i0];
                float det = d1.x * d2.y - d2.x * d1.y;
                if (Mathf.Abs(det) < 1e-12f) continue;
                float r = 1f / det;
                Vector3 sdir = (e1 * d2.y - e2 * d1.y) * r;
                Vector3 tdir = (e2 * d1.x - e1 * d2.x) * r;
                tan1[i0] += sdir; tan1[i1] += sdir; tan1[i2] += sdir;
                tan2[i0] += tdir; tan2[i1] += tdir; tan2[i2] += tdir;
            }
            var tangents = new Vector4[n];
            for (int i = 0; i < n; i++)
            {
                Vector3 nv = nrm[i], tv = tan1[i];
                Vector3 ortho = tv - nv * Vector3.Dot(nv, tv);
                if (ortho.sqrMagnitude < 1e-12f)
                    ortho = Vector3.Cross(nv, Mathf.Abs(nv.x) < 0.9f ? Vector3.right : Vector3.up);
                ortho.Normalize();
                float w = Vector3.Dot(Vector3.Cross(nv, tv), tan2[i]) < 0f ? -1f : 1f;
                tangents[i] = new Vector4(ortho.x, ortho.y, ortho.z, w);
            }
            return tangents;
        }

        void ReportStatus(string status)
        {
            StatusChanged?.Invoke(status);
        }
    }
}
