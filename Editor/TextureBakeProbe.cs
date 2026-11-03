using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// Editor tools for a saved scan package's texture (a <c>pkg_*</c> folder pulled from the device, with
    /// <c>refined_mesh.bin</c> and <c>keyframes/</c>). Choose it with Tools → Room Scan → Choose Scan Package….
    /// <list type="bullet">
    /// <item><b>Re-bake Package Atlas</b> (Play: the bake awaits frames and GPU readbacks): the package's atlas baked
    /// again from its own keyframes with the open scene's <see cref="TextureRefinement"/> settings, into
    /// <see cref="OutDir"/>/&lt;name&gt;.png (status in status.txt). <see cref="Start"/> also filters keyframes by
    /// hand footprint or head speed and can bake into a denser atlas.</item>
    /// <item><b>Render Baked Atlas From Keyframes</b> (edit mode): <see cref="RenderViews"/>.</item>
    /// </list>
    /// </summary>
    public static class TextureBakeProbe
    {
        public const string PackagePref = "QRS.TextureBakeProbe.Package";
        public const string OutDir = "Temp/TextureBake";

        [MenuItem("Tools/Room Scan/Choose Scan Package…")]
        static void MenuChoose()
        {
            string dir = EditorUtility.OpenFolderPanel("Scan package (pkg_*)", EditorPrefs.GetString(PackagePref, ""), "");
            if (string.IsNullOrEmpty(dir)) return;
            if (!File.Exists(Path.Combine(dir, "refined_mesh.bin")) || !Directory.Exists(Path.Combine(dir, "keyframes")))
            {
                EditorUtility.DisplayDialog("Scan package", "That folder has no refined_mesh.bin and keyframes/.", "OK");
                return;
            }
            EditorPrefs.SetString(PackagePref, dir);
            Debug.Log("[TextureBakeProbe] package: " + dir);
        }

        [MenuItem("Tools/Room Scan/Re-bake Package Atlas")]
        static void MenuAll() => Start(1.01f, "all");

        /// <summary>Bake the package in <see cref="PackagePref"/>, dropping frames whose hand footprint exceeds
        /// <paramref name="maxHandCoverage"/> (≥ 1 keeps all), into <see cref="OutDir"/>/<paramref name="name"/>.png.
        /// </summary>
        /// <param name="atlasScale">Bake the package's own UV layout into an atlas this many times larger each way
        /// (2 = four times the texels): what a denser atlas would show, without a new unwrap.</param>
        public static async void Start(float maxHandCoverage, string name, float maxHeadDegPerSec = 0f, float atlasScale = 1f)
        {
            Directory.CreateDirectory(OutDir);
            string status = Path.Combine(OutDir, "status.txt");
            void Status(string s)
            {
                File.AppendAllText(status, $"{DateTime.Now:HH:mm:ss} {name}: {s}\n");
                Debug.Log($"[TextureBakeProbe] {name}: {s}");
            }
            try
            {
                string pkg = EditorPrefs.GetString(PackagePref, "");
                if (!Directory.Exists(pkg)) { Status("no package at " + pkg); return; }
                if (!Application.isPlaying) { Status("enter Play first"); return; }
                var refine = UnityEngine.Object.FindAnyObjectByType<TextureRefinement>(FindObjectsInactive.Include);
                if (refine == null) { Status("no TextureRefinement in the scene"); return; }

                var mesh = ReadRefinedMesh(Path.Combine(pkg, "refined_mesh.bin"), atlasScale);
                string kfDir = FilterKeyframes(Path.Combine(pkg, "keyframes"), maxHandCoverage, maxHeadDegPerSec, name,
                    out int kept, out int total);
                Status($"baking {kept}/{total} keyframes (hand footprint ≤ {maxHandCoverage:P0}" +
                       (maxHeadDegPerSec > 0f ? $", head ≤ {maxHeadDegPerSec:0}°/s" : "") + ") on " +
                       $"{mesh.Indices.Length / 3} triangles, atlas {mesh.AtlasWidth}x{mesh.AtlasHeight}");
                float t0 = Time.realtimeSinceStartup;
                var (atlas, _) = await refine.BakeAtlasAsync(mesh, kfDir, Matrix4x4.identity);
                var tex = new Texture2D(mesh.AtlasWidth, mesh.AtlasHeight, TextureFormat.RGBA32, false);
                tex.LoadRawTextureData(atlas);
                tex.Apply(false);
                File.WriteAllBytes(Path.Combine(OutDir, name + ".png"), tex.EncodeToPNG());
                UnityEngine.Object.Destroy(tex);
                Status($"done in {Time.realtimeSinceStartup - t0:0.0} s → {OutDir}/{name}.png");
            }
            catch (Exception e)
            {
                Status("FAILED " + e.Message);
                Debug.LogException(e);
            }
        }

        static UnwrappedMeshResult ReadRefinedMesh(string path, float atlasScale = 1f)
        {
            using var r = new BinaryReader(File.OpenRead(path));
            r.ReadUInt32();
            r.ReadInt32();
            int vertCount = r.ReadInt32(), idxCount = r.ReadInt32();
            int atlasW = Mathf.RoundToInt(r.ReadInt32() * atlasScale), atlasH = Mathf.RoundToInt(r.ReadInt32() * atlasScale);
            var positions = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];
            var rawUVs = new float[vertCount * 2];
            for (int i = 0; i < vertCount; i++)
            {
                positions[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                normals[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                uvs[i] = new Vector2(r.ReadSingle(), r.ReadSingle());
                rawUVs[i * 2] = uvs[i].x * atlasW;
                rawUVs[i * 2 + 1] = uvs[i].y * atlasH;
            }
            var indices = new int[idxCount];
            for (int i = 0; i < idxCount; i++) indices[i] = r.ReadInt32();
            return new UnwrappedMeshResult
            {
                Positions = positions, Normals = normals, UVs = uvs, RawUVs = rawUVs, Indices = indices,
                AtlasWidth = atlasW, AtlasHeight = atlasH,
                OrigPositions = positions, OrigNormals = normals, OrigIndices = indices,
            };
        }

        [Serializable]
        class Frame
        {
            public int id;
            public float ts, px, py, pz, qx, qy, qz, qw, fx, fy, cx, cy, sw, sh, w, h;
            public string cap;
        }

        /// <summary>A keyframe folder holding only the frames whose hand footprint is at most
        /// <paramref name="max"/>: its own <c>frames.jsonl</c>, the images read from the original folder's.</summary>
        static string FilterKeyframes(string src, float max, float maxDegPerSec, string name, out int kept, out int total)
        {
            kept = total = 0;
            if (max >= 1f && maxDegPerSec <= 0f)
            {
                foreach (var line in File.ReadLines(Path.Combine(src, "frames.jsonl")))
                    if (line.Trim().Length > 0) { total++; kept++; }
                return src;
            }
            string dst = Path.GetFullPath(Path.Combine(OutDir, "kf_" + name));
            Directory.CreateDirectory(dst);
            string images = Path.Combine(dst, "images");
            if (!Directory.Exists(images)) FileUtil.CopyFileOrDirectory(Path.Combine(src, "images"), images);
            var sb = new StringBuilder();
            var lines = new List<string>();
            var frames = new List<Frame>();
            foreach (var line in File.ReadLines(Path.Combine(src, "frames.jsonl")))
            {
                if (line.Trim().Length == 0) continue;
                lines.Add(line);
                frames.Add(JsonUtility.FromJson<Frame>(line));
            }
            total = frames.Count;
            for (int i = 0; i < frames.Count; i++)
            {
                var f = frames[i];
                if (max < 1f && HandCoverage(f) > max) continue;
                if (maxDegPerSec > 0f && HeadSpeed(frames, i) > maxDegPerSec) continue;
                kept++;
                sb.AppendLine(lines[i]);
            }
            File.WriteAllText(Path.Combine(dst, "frames.jsonl"), sb.ToString());
            return dst;
        }

        /// <summary>How fast the head turned around frame <paramref name="i"/> (degrees per second, from the
        /// neighbouring keyframes' poses; the larger of the two sides).</summary>
        static float HeadSpeed(List<Frame> frames, int i)
        {
            float Speed(Frame a, Frame b)
            {
                float dt = Mathf.Abs(a.ts - b.ts);
                if (dt < 1e-3f) return 0f;
                var qa = new Quaternion(a.qx, a.qy, a.qz, a.qw);
                var qb = new Quaternion(b.qx, b.qy, b.qz, b.qw);
                return Quaternion.Angle(qa, qb) / dt;
            }
            float s = 0f;
            if (i > 0) s = Mathf.Max(s, Speed(frames[i], frames[i - 1]));
            if (i + 1 < frames.Count) s = Mathf.Max(s, Speed(frames[i], frames[i + 1]));
            return s;
        }

        public const string AtlasPref = "QRS.TextureBakeProbe.Atlas";
        public const string ViewsPref = "QRS.TextureBakeProbe.Views";

        [MenuItem("Tools/Room Scan/Render Baked Atlas From Keyframes")]
        static void MenuViews() => Debug.Log("[TextureBakeProbe] " + RenderViews());

        /// <summary>
        /// Render the package's refined mesh with a baked atlas (<see cref="AtlasPref"/>, default
        /// <see cref="OutDir"/>/all.png) through the cameras of chosen keyframes (<see cref="ViewsPref"/>: ids such as
        /// "3,40,113"; default every 16th), with the bake's own projection, and write photo | render | half-and-half
        /// to <see cref="OutDir"/>/views/&lt;atlas&gt;/kf_&lt;id&gt;.png. An atlas that agrees with the photos lines
        /// up with each of them; a patch textured from a photo whose pose was off shows as a ghost in the blend.
        /// views.txt has, per view, the luma correlation of render and photo at no shift and at the best whole-image
        /// shift (±48 px): a large best shift is the photo's own pose error against the rest. Edit mode is fine.
        /// </summary>
        public static string RenderViews()
        {
            string pkg = EditorPrefs.GetString(PackagePref, "");
            if (!Directory.Exists(pkg)) return "no package at " + pkg;
            string atlasPath = EditorPrefs.GetString(AtlasPref, "");
            if (string.IsNullOrEmpty(atlasPath)) atlasPath = Path.Combine(OutDir, "all.png");
            // "normals": the bare geometry, coloured by surface direction, to check mesh and pose against the photo.
            bool normals = atlasPath == "normals";
            if (!normals && !File.Exists(atlasPath)) return "no atlas at " + atlasPath;
            string kfDir = Path.Combine(pkg, "keyframes");
            var frames = new List<Frame>();
            foreach (var line in File.ReadLines(Path.Combine(kfDir, "frames.jsonl")))
                if (line.Trim().Length > 0) frames.Add(JsonUtility.FromJson<Frame>(line));
            var wanted = new HashSet<int>();
            string views = EditorPrefs.GetString(ViewsPref, "");
            // "all": every keyframe, numbers only (no sheets), for a per-photo offset table.
            bool all = views.Trim() == "all";
            foreach (var t in views.Split(','))
                if (int.TryParse(t.Trim(), out int id)) wanted.Add(id);
            if (all) foreach (var f in frames) wanted.Add(f.id);
            if (wanted.Count == 0)
                for (int i = 0; i < frames.Count; i += 16) wanted.Add(frames[i].id);

            string outDir = Path.Combine(OutDir, "views", Path.GetFileNameWithoutExtension(atlasPath));
            Directory.CreateDirectory(outDir);
            var m = ReadRefinedMesh(Path.Combine(pkg, "refined_mesh.bin"));
            var mesh = new UnityEngine.Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = m.Positions;
            mesh.uv = m.UVs;
            mesh.triangles = m.Indices;
            var atlas = new Texture2D(2, 2, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            if (normals)
            {
                atlas.SetPixel(0, 0, Color.white); atlas.SetPixel(1, 0, Color.white);
                atlas.SetPixel(0, 1, Color.white); atlas.SetPixel(1, 1, Color.white);
                atlas.Apply();
                var cols = new Color[m.Normals.Length];
                for (int i = 0; i < cols.Length; i++)
                {
                    var n = m.Normals[i].normalized;
                    cols[i] = new Color(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1f);
                }
                mesh.colors = cols;
            }
            else atlas.LoadImage(File.ReadAllBytes(atlasPath));
            var shader = normals
                ? Shader.Find("Universal Render Pipeline/Particles/Unlit")
                : Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Texture");
            var mat = new Material(shader) { mainTexture = atlas };
            if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", atlas);
            if (mat.HasProperty("_Cull")) mat.SetFloat("_Cull", 0f);
            const int layer = 31;
            var go = new GameObject("[AtlasView]") { hideFlags = HideFlags.HideAndDontSave, layer = layer };
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            var camGo = new GameObject("[AtlasViewCam]") { hideFlags = HideFlags.HideAndDontSave };
            var cam = camGo.AddComponent<Camera>();
            cam.enabled = false;
            cam.cullingMask = 1 << layer;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
            cam.allowMSAA = false;
            cam.allowHDR = false;
            const float near = 0.02f, far = 30f;
            cam.nearClipPlane = near;
            cam.farClipPlane = far;
            bool asyncWas = ShaderUtil.allowAsyncCompilation;
            ShaderUtil.allowAsyncCompilation = false;
            // A variant still compiling draws magenta and Camera.Render does not wait for it.
            for (int i = 0; i < mat.passCount; i++) ShaderUtil.CompilePass(mat, i, true);
            var report = new StringBuilder($"# Atlas views {DateTime.Now:yyyy-MM-dd HH:mm}: {atlasPath}\n" +
                                           "id   ncc0   nccBest  shift(px)  covered\n");
            var csv = new StringBuilder("id,ts,ncc0,nccBest,shiftX,shiftY,covered\n");
            int done = 0;
            try
            {
                foreach (var f in frames)
                {
                    if (!wanted.Contains(f.id)) continue;
                    string img = Path.Combine(kfDir, "images", $"{f.id:D6}.jpg");
                    if (!File.Exists(img)) continue;
                    int w = Mathf.RoundToInt(f.w), h = Mathf.RoundToInt(f.h);
                    // The bake's projection (TextureRefinement.ImageIntrinsics).
                    Vector4 ic = TextureRefinement.ImageIntrinsics(new Vector2(f.fx, f.fy), new Vector2(f.cx, f.cy),
                        new Vector2(f.sw, f.sh), new Vector2(f.w, f.h));
                    var proj = Matrix4x4.zero;
                    proj[0, 0] = 2f * ic.x / w;
                    proj[0, 2] = 1f - 2f * ic.z / w;
                    proj[1, 1] = 2f * ic.y / h;
                    proj[1, 2] = 1f - 2f * ic.w / h;
                    proj[2, 2] = -(far + near) / (far - near);
                    proj[2, 3] = -2f * far * near / (far - near);
                    proj[3, 2] = -1f;
                    cam.transform.SetPositionAndRotation(new Vector3(f.px, f.py, f.pz), new Quaternion(f.qx, f.qy, f.qz, f.qw));
                    cam.projectionMatrix = proj;
                    var rt = RenderTexture.GetTemporary(w, h, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                    cam.targetTexture = rt;
                    cam.Render();
                    cam.Render();   // the first submit after a material is made can still be magenta
                    cam.targetTexture = null;
                    var render = ReadBack(rt, w, h);
                    var photoTex = new Texture2D(2, 2);
                    photoTex.LoadImage(File.ReadAllBytes(img));
                    Graphics.Blit(photoTex, rt);
                    var photo = ReadBack(rt, w, h);
                    UnityEngine.Object.DestroyImmediate(photoTex);
                    RenderTexture.ReleaseTemporary(rt);

                    Compare(photo, render, w, h, out float ncc0, out float nccBest, out Vector2Int shift, out float covered);
                    report.AppendLine($"{f.id,3}  {ncc0,5:F2}  {nccBest,6:F2}  ({shift.x,4}, {shift.y,4})  {covered:P0}");
                    csv.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0},{1:F3},{2:F3},{3:F3},{4},{5},{6:F3}",
                        f.id, f.ts, ncc0, nccBest, shift.x, shift.y, covered));
                    if (all)
                    {
                        // The render alone at half size, for offline comparison with the photo.
                        var half = RenderTexture.GetTemporary(w / 2, h / 2, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                        Graphics.Blit(render, half);
                        var small = ReadBack(half, w / 2, h / 2);
                        RenderTexture.ReleaseTemporary(half);
                        File.WriteAllBytes(Path.Combine(outDir, $"r_{f.id:D3}.png"), small.EncodeToPNG());
                        UnityEngine.Object.DestroyImmediate(small);
                        UnityEngine.Object.DestroyImmediate(photo);
                        UnityEngine.Object.DestroyImmediate(render);
                        done++;
                        continue;
                    }
                    var sheet = new Texture2D(w * 3, h, TextureFormat.RGBA32, false);
                    var dst = sheet.GetPixelData<Color32>(0);
                    var a = photo.GetPixelData<Color32>(0);
                    var b = render.GetPixelData<Color32>(0);
                    for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int i = y * w + x, o = y * w * 3 + x;
                        Color32 pa = a[i], pb = b[i];
                        dst[o] = pa;
                        dst[o + w] = pb;
                        dst[o + 2 * w] = new Color32((byte)((pa.r + pb.r) / 2), (byte)((pa.g + pb.g) / 2), (byte)((pa.b + pb.b) / 2), 255);
                    }
                    sheet.Apply(false);
                    File.WriteAllBytes(Path.Combine(outDir, $"kf_{f.id:D3}.png"), sheet.EncodeToPNG());
                    UnityEngine.Object.DestroyImmediate(sheet);
                    UnityEngine.Object.DestroyImmediate(photo);
                    UnityEngine.Object.DestroyImmediate(render);
                    done++;
                }
            }
            finally
            {
                ShaderUtil.allowAsyncCompilation = asyncWas;
                UnityEngine.Object.DestroyImmediate(camGo);
                UnityEngine.Object.DestroyImmediate(go);
                UnityEngine.Object.DestroyImmediate(mat);
                UnityEngine.Object.DestroyImmediate(atlas);
                UnityEngine.Object.DestroyImmediate(mesh);
            }
            File.WriteAllText(Path.Combine(outDir, "views.txt"), report.ToString());
            File.WriteAllText(Path.Combine(outDir, "views.csv"), csv.ToString());
            return $"{done} views → {outDir}";
        }

        static Texture2D ReadBack(RenderTexture rt, int w, int h)
        {
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var t = new Texture2D(w, h, TextureFormat.RGBA32, false);
            t.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            t.Apply(false);
            RenderTexture.active = prev;
            return t;
        }

        /// <summary>Luma correlation of photo and render over the rendered pixels, at no shift and at the best
        /// whole-image shift of the photo against the render within ±96 px (searched at 1/8 resolution, refined at
        /// 1/4; reported in full pixels, + = the photo's content sits right / up of the render's).</summary>
        static void Compare(Texture2D photo, Texture2D render, int w, int h, out float ncc0, out float best,
            out Vector2Int shift, out float covered)
        {
            var a = photo.GetPixelData<Color32>(0);
            var b = render.GetPixelData<Color32>(0);
            void Level(int s, out float[] pl, out float[] rl, out bool[] rm, out int lw, out int lh)
            {
                lw = w / s; lh = h / s;
                pl = new float[lw * lh]; rl = new float[lw * lh]; rm = new bool[lw * lh];
                for (int y = 0; y < lh; y++)
                for (int x = 0; x < lw; x++)
                {
                    float sa = 0, sb = 0;
                    int cov = 0;
                    for (int dy = 0; dy < s; dy++)
                    for (int dx = 0; dx < s; dx++)
                    {
                        int i = (y * s + dy) * w + x * s + dx;
                        Color32 ca = a[i], cb = b[i];
                        sa += 0.3f * ca.r + 0.59f * ca.g + 0.11f * ca.b;
                        sb += 0.3f * cb.r + 0.59f * cb.g + 0.11f * cb.b;
                        if (cb.r + cb.g + cb.b > 6) cov++;
                    }
                    pl[y * lw + x] = sa / (s * s);
                    rl[y * lw + x] = sb / (s * s);
                    rm[y * lw + x] = cov == s * s;
                }
            }
            float Ncc(float[] pl, float[] rl, bool[] rm, int lw, int lh, int margin, int sx, int sy)
            {
                double sp = 0, sr = 0, spp = 0, srr = 0, spr = 0;
                int n = 0;
                for (int y = margin; y < lh - margin; y++)
                for (int x = margin; x < lw - margin; x++)
                {
                    int i = y * lw + x;
                    if (!rm[i]) continue;
                    float r = rl[i], p = pl[(y + sy) * lw + x + sx];
                    sp += p; sr += r; spp += p * p; srr += r * r; spr += p * r;
                    n++;
                }
                if (n < 100) return -1f;
                double vp = spp - sp * sp / n, vr = srr - sr * sr / n;
                return vp <= 1e-6 || vr <= 1e-6 ? -1f : (float)((spr - sp * sr / n) / Math.Sqrt(vp * vr));
            }
            Level(8, out var p8, out var r8, out var m8, out int w8, out int h8);
            Level(4, out var p4, out var r4, out var m4, out int w4, out int h4);
            int c = 0;
            foreach (bool v in m4) if (v) c++;
            covered = (float)c / m4.Length;
            const int r8max = 12;   // ±96 px
            int bx = 0, by = 0;
            float b8 = -2f;
            for (int sy = -r8max; sy <= r8max; sy++)
            for (int sx = -r8max; sx <= r8max; sx++)
            {
                float v = Ncc(p8, r8, m8, w8, h8, r8max, sx, sy);
                if (v > b8) { b8 = v; bx = sx; by = sy; }
            }
            const int margin4 = 2 * r8max + 2;
            ncc0 = Ncc(p4, r4, m4, w4, h4, margin4, 0, 0);
            best = -2f;
            shift = Vector2Int.zero;
            for (int sy = 2 * by - 2; sy <= 2 * by + 2; sy++)
            for (int sx = 2 * bx - 2; sx <= 2 * bx + 2; sx++)
            {
                float v = Ncc(p4, r4, m4, w4, h4, margin4, sx, sy);
                if (v > best) { best = v; shift = new Vector2Int(sx * 4, sy * 4); }
            }
        }

        /// <summary>The collector's estimate (<c>KeyframeCollector.HandCoverage</c>) on a recorded frame, with the
        /// focal length scaled to the saved image.</summary>
        public static float HandCoverageOf(string framesJsonlLine) => HandCoverage(JsonUtility.FromJson<Frame>(framesJsonlLine));

        static float HandCoverage(Frame f)
        {
            if (string.IsNullOrEmpty(f.cap) || f.w <= 1f || f.h <= 1f) return 0f;
            var camPos = new Vector3(f.px, f.py, f.pz);
            var inv = Quaternion.Inverse(new Quaternion(f.qx, f.qy, f.qz, f.qw));
            Vector4 ic = TextureRefinement.ImageIntrinsics(new Vector2(f.fx, f.fy), new Vector2(f.cx, f.cy),
                new Vector2(f.sw, f.sh), new Vector2(f.w, f.h));
            var focal = new Vector2(ic.x, ic.y);
            var principal = new Vector2(ic.z, ic.w);
            var res = new Vector2(f.w, f.h);
            float total = 0f;
            foreach (var c in f.cap.Split(';'))
            {
                var v = c.Trim().Split(' ');
                if (v.Length < 7) continue;
                float P(int i) => float.Parse(v[i], CultureInfo.InvariantCulture);
                Vector3 a = inv * (new Vector3(P(0), P(1), P(2)) - camPos);
                Vector3 b = inv * (new Vector3(P(3), P(4), P(5)) - camPos);
                float rad = P(6);
                const float near = 0.05f;
                if (a.z < near && b.z < near) continue;
                if (a.z < near) a = Vector3.Lerp(a, b, (near - a.z) / (b.z - a.z));
                else if (b.z < near) b = Vector3.Lerp(b, a, (near - b.z) / (a.z - b.z));
                Vector2 pa = new Vector2(focal.x * a.x / a.z, focal.y * a.y / a.z) + principal;
                Vector2 pb = new Vector2(focal.x * b.x / b.z, focal.y * b.y / b.z) + principal;
                float pr = focal.x * rad / Mathf.Min(a.z, b.z);
                Vector2 min = Vector2.Max(Vector2.Min(pa, pb) - Vector2.one * pr, Vector2.zero);
                Vector2 maxv = Vector2.Min(Vector2.Max(pa, pb) + Vector2.one * pr, res);
                if (maxv.x <= min.x || maxv.y <= min.y) continue;
                float len = Vector2.Distance(pa, pb);
                total += Mathf.Min(len * 2f * pr + Mathf.PI * pr * pr, (maxv.x - min.x) * (maxv.y - min.y));
            }
            return Mathf.Clamp01(total / (res.x * res.y));
        }
    }
}
