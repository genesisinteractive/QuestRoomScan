using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// Drives a <see cref="KeyframeCollector"/> with a synthetic head turn in Play to check how it treats motion: a
    /// fast sweep (80 deg/s) across 120° of yaw, then a slow sweep (8 deg/s) back over the same view. The fast pass
    /// saves smeared frames; the slow pass must save steady retakes over them (once per pose), and every keyframe must
    /// carry its head speed ("av") for the bake's sharpness weight.
    /// Menu: Tools → Room Scan → Check Keyframe Steady Retakes (Play). Report: <see cref="ReportPath"/>.
    /// </summary>
    public static class KeyframeRetakeProbe
    {
        public const string ReportPath = "Temp/KeyframeRetake/report.txt";
        const string ExportDir = "Temp/KeyframeRetake/keyframes";

        [MenuItem("Tools/Room Scan/Check Keyframe Steady Retakes (Play)")]
        static void Menu() => Debug.Log("[KeyframeRetakeProbe] " + Start());

        static GameObject _host;
        static KeyframeCollector _collector;
        static Texture2D _frame;
        static double _t0;
        static int _step;

        public static string Start()
        {
            if (!Application.isPlaying) return "enter Play first";
            if (_host != null) return "already running";
            Directory.CreateDirectory(Path.GetDirectoryName(ReportPath));
            File.WriteAllText(ReportPath, "running\n");
            string dir = Path.GetFullPath(ExportDir);
            if (Directory.Exists(dir)) Directory.Delete(dir, true);
            _host = new GameObject("[KeyframeRetakeProbe]") { hideFlags = HideFlags.DontSave };
            _collector = _host.AddComponent<KeyframeCollector>();
            _collector.SetExportDirectory(dir);
            _frame = new Texture2D(64, 48, TextureFormat.RGBA32, false);
            _step = 0;
            EditorApplication.update += Tick;
            return "running about 18 s; report at " + ReportPath;
        }

        const float FastDegPerSec = 80f, SlowDegPerSec = 8f, SpanDeg = 120f;

        static void Tick()
        {
            if (!Application.isPlaying || _collector == null) { Stop("Play stopped"); return; }
            // Start has to have run (it arms the collector) before the first frame.
            if (_step++ < 2) { _t0 = Time.realtimeSinceStartupAsDouble; return; }
            double t = Time.realtimeSinceStartupAsDouble - _t0;
            double fastEnd = SpanDeg / FastDegPerSec, slowEnd = fastEnd + 0.5 + SpanDeg / SlowDegPerSec;
            float yaw;
            if (t < fastEnd) yaw = (float)(t * FastDegPerSec);
            else if (t < fastEnd + 0.5) yaw = SpanDeg;                 // a beat at the end of the fast sweep
            else if (t < slowEnd) yaw = SpanDeg - (float)((t - fastEnd - 0.5) * SlowDegPerSec);
            else
            {
                if (t > slowEnd + 1.0) Finish(fastEnd);               // let the writes land
                return;
            }
            var rot = Quaternion.Euler(0f, yaw, 0f);
            _collector.TrySaveKeyframe(_frame, new Vector3(0f, 1.6f, 0f), rot, new Vector2(50f, 50f),
                new Vector2(32f, 24f), new Vector2(64f, 48f), new Vector2(64f, 48f));
        }

        [Serializable]
        class Row { public int id; public float ts; public float av = -1f; public float qy, qw; }

        static void Finish(double fastEnd)
        {
            string manifest = Path.Combine(Path.GetFullPath(ExportDir), "frames.jsonl");
            var rows = File.Exists(manifest)
                ? File.ReadAllLines(manifest).Where(l => l.Trim().Length > 0).Select(JsonUtility.FromJson<Row>)
                    .OrderBy(r => r.id).ToList()
                : new System.Collections.Generic.List<Row>();
            int fast = rows.Count(r => r.av >= 30f), steady = rows.Count(r => r.av >= 0f && r.av <= 15f);
            int noAv = rows.Count(r => r.av < 0f);
            var sb = new System.Text.StringBuilder($"# Keyframe steady retakes {DateTime.Now:yyyy-MM-dd HH:mm}\n");
            sb.AppendLine($"fast sweep {FastDegPerSec} deg/s then slow sweep {SlowDegPerSec} deg/s back over {SpanDeg} deg");
            sb.AppendLine($"{rows.Count} keyframes: {fast} at ≥ 30 deg/s, {steady} at ≤ 15 deg/s, {noAv} without av; " +
                          $"collector counted {_collector.SharpRetakes} retakes");
            float Yaw(Row r) => 2f * Mathf.Atan2(r.qy, r.qw) * Mathf.Rad2Deg;
            int fastWithSteady = 0;
            foreach (var r in rows)
            {
                bool isFast = r.av >= 30f;
                bool near = isFast && rows.Any(o => o.av >= 0f && o.av <= 15f && Mathf.Abs(Mathf.DeltaAngle(Yaw(o), Yaw(r))) < 10f);
                if (near) fastWithSteady++;
                sb.AppendLine($"  {r.id,3}  yaw {Yaw(r),6:F1}  av {r.av,6:F1}{(isFast ? near ? "  steady view within 10°" : "  NO steady view within 10°" : "")}");
            }
            sb.AppendLine($"{fastWithSteady}/{fast} fast keyframes have a steady one within 10°");
            // The slow sweep's frames (after the beat at the end of the fast one) must read its speed.
            var slow = rows.Where(r => r.av >= 0f && r.av <= 15f && Yaw(r) < SpanDeg - 4f).Select(r => r.av).OrderBy(v => v).ToList();
            float slowMedian = slow.Count > 0 ? slow[slow.Count / 2] : -1f;
            bool slowRead = slowMedian > SlowDegPerSec * 0.6f && slowMedian < SlowDegPerSec * 1.4f;
            sb.AppendLine($"slow sweep reads {slowMedian:F1} deg/s (driven at {SlowDegPerSec})");
            // Pass: the fast sweep was recorded fast; every pose it saved got a steady view in the slow sweep;
            // the retakes stay bounded (at most about two steady frames per fast one: the retake and a new
            // standing yaw between two fast frames); only the very first frame (nothing to measure from) lacks a speed.
            bool ok = fast >= 4 && fastWithSteady == fast && steady <= 2 * fast + 2 && noAv <= 1 &&
                      _collector.SharpRetakes > 0 && slowRead;
            sb.AppendLine(ok ? "PASS" : "FAIL");
            File.WriteAllText(ReportPath, sb.ToString());
            Debug.Log("[KeyframeRetakeProbe]\n" + sb);
            Stop(null);
        }

        static void Stop(string why)
        {
            EditorApplication.update -= Tick;
            if (why != null) File.AppendAllText(ReportPath, why + "\n");
            if (_host != null) UnityEngine.Object.Destroy(_host);
            if (_frame != null) UnityEngine.Object.Destroy(_frame);
            _host = null;
            _collector = null;
            _frame = null;
        }
    }
}
