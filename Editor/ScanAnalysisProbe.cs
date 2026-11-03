using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// Edit-mode check of the scan analysis (closure, surface, leaks) on a synthetic room: a box written straight into
    /// a real <see cref="VolumeIntegrator"/>'s TSDF, one analysis cycle run on the GPU, and its surface / leak numbers
    /// compared with a CPU count of the same faces on the same voxels. No headset, no depth.
    /// Menu: Tools → Room Scan → Check Scan Analysis.
    /// </summary>
    public static class ScanAnalysisProbe
    {
        const string ComputePath = "Packages/com.genesis.roomscan/Runtime/Shaders/VolumeIntegration.compute";

        /// <summary>One synthetic room.</summary>
        public struct Room
        {
            public string Name;
            /// <summary>Room box (metres, world = volume space).</summary>
            public Vector3 Min, Max;
            /// <summary>How deep behind the walls the depth wrote (metres).</summary>
            public float BackBand;
            /// <summary>Weight of the solid band behind the walls (the free side gets <see cref="FreeWeight"/>).</summary>
            public float SolidWeight, FreeWeight;
            /// <summary>Freeze (negate the weight of) every observed voxel.</summary>
            public bool Frozen;
            /// <summary>A square hole in the +z wall, this wide (metres; 0 none).</summary>
            public float Hole;
            /// <summary>The room turned about the vertical (degrees), so its walls cross the block grid.</summary>
            public float Yaw;
            /// <summary>Override the normal hole cap (m²; 0 keeps the integrator's).</summary>
            public float FillCap;
        }

        [MenuItem("Tools/Room Scan/Check Scan Analysis")]
        static void Menu() => Debug.Log(RunDefaults());

        /// <summary>The standard set: a closed room on and off the 8-voxel block grid, frozen, with a thin solid band,
        /// with a hole.</summary>
        public static string RunDefaults()
        {
            var sb = new StringBuilder("[ScanAnalysisProbe]\n");
            var box = new Room
            {
                Name = "closed, walls on block edges", Min = new Vector3(-2f, -1.2f, -1.6f), Max = new Vector3(2f, 1.2f, 1.6f),
                BackBand = 0.3f, SolidWeight = 0.4f, FreeWeight = 0.4f,
            };
            sb.AppendLine(Run(box));
            var off = box;
            off.Name = "closed, walls off block edges";
            off.Min += new Vector3(0.12f, 0.07f, 0.17f);
            off.Max += new Vector3(0.12f, 0.07f, 0.17f);
            sb.AppendLine(Run(off));
            var frozen = off;
            frozen.Name = "closed, all frozen";
            frozen.Frozen = true;
            sb.AppendLine(Run(frozen));
            var thin = off;
            thin.Name = "closed, solid band one voxel";
            thin.BackBand = 0.05f;
            sb.AppendLine(Run(thin));
            var weak = off;
            weak.Name = "closed, solid band at seed weight";
            weak.SolidWeight = 0.1f;
            sb.AppendLine(Run(weak));
            var hole = off;
            hole.Name = "0.5 m hole in a wall";
            hole.Hole = 0.5f;
            sb.AppendLine(Run(hole));
            var turned = off;
            turned.Name = "closed, turned 20°";
            turned.Yaw = 20f;
            sb.AppendLine(Run(turned));
            var floor = off;
            floor.Name = "closed, floor at y = 0 (the headset's floor: a block edge)";
            floor.Min.y = 0f;
            floor.Max.y = 2.5f;
            sb.AppendLine(Run(floor));
            var floorHole = floor;
            floorHole.Name = "floor at y = 0, 0.5 m hole in a wall";
            floorHole.Hole = 0.5f;
            sb.AppendLine(Run(floorHole));
            var tiny = floor;
            tiny.Name = "0.15 m hole, three cycles";
            tiny.Hole = 0.15f;
            sb.AppendLine(Run(tiny, 128, 3, -1f));
            var small = floor;
            small.Name = "0.5 m hole, three cycles (the normal cap)";
            small.Hole = 0.5f;
            sb.AppendLine(Run(small, 128, 3, -1f));
            var window = floor;
            window.Name = "a 0.8 m window depth cannot close, stall capping";
            window.Hole = 0.8f;
            sb.AppendLine(Run(window, 128, 4, 0f));
            var windowOff = window;
            windowOff.Name = "the same window, no stall yet (20 s clock)";
            sb.AppendLine(Run(windowOff, 128, 4, 20f));
            return sb.ToString();
        }

        public static string Run(Room room, int n = 128) => Run(room, n, 1, -1f);

        /// <summary><paramref name="cycles"/> analysis cycles in a row (the caps of one cycle show in the next);
        /// <paramref name="stallSeconds"/> ≥ 0 overrides the stall clock.</summary>
        public static string Run(Room room, int n, int cycles, float stallSeconds)
        {
            if (EditorApplication.isPlaying) return "stop Play first";
            var compute = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
            if (compute == null)
            {
                foreach (var g in AssetDatabase.FindAssets("VolumeIntegration t:ComputeShader"))
                {
                    compute = AssetDatabase.LoadAssetAtPath<ComputeShader>(AssetDatabase.GUIDToAssetPath(g));
                    if (compute != null) break;
                }
            }
            if (compute == null) return "VolumeIntegration.compute not found";

            var go = new GameObject("ScanAnalysisProbe") { hideFlags = HideFlags.HideAndDontSave };
            var vi = go.AddComponent<VolumeIntegrator>();
            try
            {
                var so = new SerializedObject(vi);
                so.FindProperty("compute").objectReferenceValue = compute;
                var vc = so.FindProperty("voxelCount");
                vc.FindPropertyRelative("x").intValue = n;
                vc.FindPropertyRelative("y").intValue = n;
                vc.FindPropertyRelative("z").intValue = n;
                if (stallSeconds >= 0f) so.FindProperty("stallSeconds").floatValue = stallSeconds;
                bool fill = so.FindProperty("fillLeaks").boolValue;
                if (room.FillCap > 0f) so.FindProperty("fillLeakMaxAreaM2").floatValue = room.FillCap;
                so.ApplyModifiedPropertiesWithoutUndo();
                so.ApplyModifiedPropertiesWithoutUndo();
                float voxel = so.FindProperty("voxelSize").floatValue;
                float trunc = so.FindProperty("voxelDistance").floatValue;
                float minMesh = so.FindProperty("minMeshWeight").floatValue;

                vi.ReallocateVolumes();
                var data = Build(room, n, voxel, trunc, out var cpu, minMesh);
                var tex = new Texture3D(n, n, n, GraphicsFormat.R8G8_SNorm, TextureCreationFlags.None);
                tex.SetPixelData(data, 0);
                tex.Apply(false);
                Graphics.CopyTexture(tex, vi.Volume);
                Object.DestroyImmediate(tex);

                vi.ResetAnalysis();
                var leaks = new StringBuilder();
                for (int cycle = 1; cycle <= cycles; cycle++)
                {
                    vi.RequestAnalysisNow();
                    for (int i = 0; i < 400 && vi.AnalysisCycles < cycle; i++)
                    {
                        vi.StepAnalysis();
                        AsyncGPUReadback.WaitAllRequests();
                    }
                    if (vi.AnalysisCycles < cycle) return $"{room.Name}: no analysis result (cycle {cycle})";
                    if (cycles > 1)
                        leaks.Append($" [{cycle}: leak {vi.Closure.LeakAreaM2:0.00} m²{(vi.LeakStalled ? " stalled" : "")} " +
                                     $"fills {vi.LeakFills}]");
                }
                var c = vi.Closure;
                float face = voxel * voxel;
                return $"{room.Name}: GPU surface {c.SurfaceAreaM2:0.00} m² (faces {vi.FaceSurfaceAreaM2:0.00}, band " +
                       $"{vi.BandSurfaceAreaM2:0.00}) leak {c.LeakAreaM2:0.00} m² closure {c.Closure:P1} holes {c.HoleCount} | " +
                       $"CPU F–S {cpu.FS * face:0.00} m² F–E {cpu.FE * face:0.00} m² F–U(sealed) {cpu.FU * face:0.00} m² " +
                       $"(GPU {vi.SealedUnknownFaces * face:0.00}) | box {cpu.BoxArea:0.00} m² | surface voxels GPU " +
                       $"{vi.SurfaceVoxelCount} CPU {cpu.SurfaceVoxels}{(fill ? "" : " (fill off)")}{leaks}";
            }
            finally
            {
                vi.ReleaseAll();
                Object.DestroyImmediate(go);
            }
        }

        struct Cpu
        {
            public int FS, FE, FU, SurfaceVoxels;
            public float BoxArea;
        }

        static float Dist(Vector3 p, Room r)
        {
            // signed distance to the box: + inside
            Vector3 c = (r.Min + r.Max) * 0.5f, h = (r.Max - r.Min) * 0.5f;
            Vector3 q = new(Mathf.Abs(p.x - c.x) - h.x, Mathf.Abs(p.y - c.y) - h.y, Mathf.Abs(p.z - c.z) - h.z);
            float outside = new Vector3(Mathf.Max(q.x, 0f), Mathf.Max(q.y, 0f), Mathf.Max(q.z, 0f)).magnitude;
            float inside = Mathf.Min(Mathf.Max(q.x, Mathf.Max(q.y, q.z)), 0f);
            return -(outside + inside);
        }

        static byte[] Build(Room r, int n, float voxel, float trunc, out Cpu cpu, float minMesh)
        {
            var data = new byte[n * n * n * 2];
            var tsdf = new float[n * n * n];
            var w = new float[n * n * n];
            float half = n * 0.5f;
            for (int z = 0; z < n; z++)
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                var p = new Vector3((x + 0.5f - half) * voxel, (y + 0.5f - half) * voxel, (z + 0.5f - half) * voxel);
                if (r.Yaw != 0f)
                {
                    Vector3 c = (r.Min + r.Max) * 0.5f;
                    p = c + Quaternion.AngleAxis(-r.Yaw, Vector3.up) * (p - c);
                }
                float d = Dist(p, r);
                bool inHole = r.Hole > 0f && p.z > r.Max.z - 0.4f && Mathf.Abs(p.x - (r.Min.x + r.Max.x) * 0.5f) < r.Hole * 0.5f
                              && Mathf.Abs(p.y - (r.Min.y + r.Max.y) * 0.5f) < r.Hole * 0.5f;
                float t, wt;
                if (d >= 0f)
                {
                    t = Mathf.Clamp(d / trunc, -1f, 1f);
                    wt = r.FreeWeight;
                }
                else if (d >= -r.BackBand && !inHole)
                {
                    t = Mathf.Clamp(d / trunc, -1f, 1f);
                    wt = r.SolidWeight;
                }
                else
                {
                    t = -1f;
                    wt = 0f;
                }
                if (r.Frozen) wt = -wt;
                int i = x + y * n + z * n * n;
                tsdf[i] = t;
                w[i] = wt;
                data[i * 2] = (byte)(sbyte)Mathf.RoundToInt(t * 127f);
                data[i * 2 + 1] = (byte)(sbyte)Mathf.RoundToInt(wt * 127f);
            }

            // the analysis' own classes, from the stored (quantised) values
            cpu = default;
            var lbl = new byte[n * n * n]; // 0 F, 1 S, 2 U
            for (int i = 0; i < lbl.Length; i++)
            {
                float t = (sbyte)data[i * 2] / 127f, wt = Mathf.Abs((sbyte)data[i * 2 + 1] / 127f);
                lbl[i] = (byte)(wt < minMesh ? 2 : t > 0f ? 0 : 1);
                if (wt > 0.05f && Mathf.Abs(t) < 0.3f) cpu.SurfaceVoxels++;
            }
            // exterior: U voxels connected to the volume's edge (6-flood); the GPU also counts U in a pure-U 8³ block
            var ext = new bool[lbl.Length];
            var stack = new System.Collections.Generic.Stack<int>();
            for (int z = 0; z < n; z++)
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                if (x != 0 && y != 0 && z != 0 && x != n - 1 && y != n - 1 && z != n - 1) continue;
                int i = x + y * n + z * n * n;
                if (lbl[i] == 2 && !ext[i]) { ext[i] = true; stack.Push(i); }
            }
            while (stack.Count > 0)
            {
                int i = stack.Pop();
                int x = i % n, y = i / n % n, z = i / (n * n);
                void Try(int xx, int yy, int zz)
                {
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= n || yy >= n || zz >= n) return;
                    int j = xx + yy * n + zz * n * n;
                    if (lbl[j] != 2 || ext[j]) return;
                    ext[j] = true;
                    stack.Push(j);
                }
                Try(x + 1, y, z); Try(x - 1, y, z); Try(x, y + 1, z); Try(x, y - 1, z); Try(x, y, z + 1); Try(x, y, z - 1);
            }
            for (int z = 1; z < n - 1; z++)
            for (int y = 1; y < n - 1; y++)
            for (int x = 1; x < n - 1; x++)
            {
                int i = x + y * n + z * n * n;
                if (lbl[i] != 0) continue;
                int[] nb = { i + 1, i - 1, i + n, i - n, i + n * n, i - n * n };
                foreach (int j in nb)
                {
                    if (lbl[j] == 1) cpu.FS++;
                    else if (lbl[j] == 2) { if (ext[j]) cpu.FE++; else cpu.FU++; }
                }
            }
            Vector3 s = r.Max - r.Min;
            cpu.BoxArea = 2f * (s.x * s.y + s.y * s.z + s.x * s.z);
            return data;
        }
    }
}
