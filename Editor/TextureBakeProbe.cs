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
    /// Re-bake a saved scan package's texture atlas in the editor from its own keyframes and refined mesh, the way
    /// the headset bakes it (the same <see cref="TextureRefinement"/> component and settings as the open scene), to
    /// compare bakes on one real scan: all keyframes, or leaving out frames whose hand footprint (the recorded hand /
    /// forearm capsules projected into the frame) is over a fraction of the image.
    /// <para>Set the package folder (a pulled <c>pkg_*</c> with <c>refined_mesh.bin</c> and <c>keyframes/</c>) in
    /// <see cref="PackagePref"/>, enter Play (the bake awaits frames and GPU readbacks), then
    /// Tools → Room Scan → Re-bake Package Atlas (…). Results: <see cref="OutDir"/>/&lt;name&gt;.png and status.txt.
    /// </para>
    /// </summary>
    public static class TextureBakeProbe
    {
        public const string PackagePref = "QRS.TextureBakeProbe.Package";
        public const string OutDir = "Temp/TextureBake";

        [MenuItem("Tools/Room Scan/Re-bake Package Atlas (all keyframes)")]
        static void MenuAll() => Start(1.01f, "all");

        [MenuItem("Tools/Room Scan/Re-bake Package Atlas (hands at most 12 %)")]
        static void MenuTwelve() => Start(0.12f, "hands12");

        [MenuItem("Tools/Room Scan/Re-bake Package Atlas (no hands in view)")]
        static void MenuNone() => Start(0.0001f, "nohands");

        [MenuItem("Tools/Room Scan/Re-bake Package Atlas (head under 25 deg per s)")]
        static void MenuSlow() => Start(1.01f, "slow25", 25f);

        [MenuItem("Tools/Room Scan/Re-bake Package Atlas (head under 25 deg per s, hands at most 12 %)")]
        static void MenuSlowHands() => Start(0.12f, "slow25_hands12", 25f);

        /// <summary>Bake the package in <see cref="PackagePref"/>, dropping frames whose hand footprint exceeds
        /// <paramref name="maxHandCoverage"/> (≥ 1 keeps all), into <see cref="OutDir"/>/<paramref name="name"/>.png.
        /// </summary>
        public static async void Start(float maxHandCoverage, string name, float maxHeadDegPerSec = 0f)
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

                var mesh = ReadRefinedMesh(Path.Combine(pkg, "refined_mesh.bin"));
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

        static UnwrappedMeshResult ReadRefinedMesh(string path)
        {
            using var r = new BinaryReader(File.OpenRead(path));
            r.ReadUInt32();
            r.ReadInt32();
            int vertCount = r.ReadInt32(), idxCount = r.ReadInt32();
            int atlasW = r.ReadInt32(), atlasH = r.ReadInt32();
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
            public float ts, px, py, pz, qx, qy, qz, qw, fx, fy, sw, sh, w, h;
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

        /// <summary>The collector's estimate (<c>KeyframeCollector.HandCoverage</c>) on a recorded frame, with the
        /// focal length scaled to the saved image.</summary>
        public static float HandCoverageOf(string framesJsonlLine) => HandCoverage(JsonUtility.FromJson<Frame>(framesJsonlLine));

        static float HandCoverage(Frame f)
        {
            if (string.IsNullOrEmpty(f.cap) || f.w <= 1f || f.h <= 1f) return 0f;
            var camPos = new Vector3(f.px, f.py, f.pz);
            var inv = Quaternion.Inverse(new Quaternion(f.qx, f.qy, f.qz, f.qw));
            float scale = f.sw > 1f ? f.w / f.sw : 1f;
            var focal = new Vector2(f.fx, f.fy) * scale;
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
                Vector2 pa = new Vector2(focal.x * a.x / a.z, focal.y * a.y / a.z) + res * 0.5f;
                Vector2 pb = new Vector2(focal.x * b.x / b.z, focal.y * b.y / b.z) + res * 0.5f;
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
