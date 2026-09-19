using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Timing ledger for one texture refinement, written to the log as a
    /// single <c>[TextureRefine][Profile]</c> block so a headset run can be
    /// diagnosed from logcat alone. Three views of the same bake:
    ///
    /// <list type="bullet">
    /// <item><b>Stages</b> — wall time per pipeline stage (unwrap, pass 1,
    /// pass 2, seam levelling, …), from <see cref="Stage"/>.</item>
    /// <item><b>Keyframes</b> — where the main thread spent its time per
    /// keyframe: waiting on the worker decode, uploading pixels, the
    /// registration hook (which awaits a GPU readback), issuing dispatches;
    /// and what the worker spent reading and decoding.</item>
    /// <item><b>Frames</b> — every compositor frame the bake ran across, from
    /// <c>Time.unscaledDeltaTime</c> at each continuation: count, mean, max,
    /// how many missed 72 Hz and how many were outright hitches. This is the
    /// number the player feels.</item>
    /// </list>
    ///
    /// Everything is a <see cref="Stopwatch"/> read or an addition; the
    /// profiler itself costs nothing a frame can notice.
    /// </summary>
    internal sealed class RefineProfile
    {
        public const float FrameBudgetMs = 1000f / 72f;
        public const float HitchMs = 25f;

        readonly Stopwatch _clock = Stopwatch.StartNew();
        readonly List<(string name, double ms)> _stages = new();
        readonly long _startManaged, _startNative;

        // Per-keyframe main-thread buckets.
        double _decodeWait, _upload, _refine, _issue;
        double _decodeWaitMax, _uploadMax, _refineMax, _issueMax;
        // Worker buckets, reported by the decoder.
        double _read, _decode, _readMax, _decodeMax;
        int _keyframes, _decodeFallbacks;

        // Frame buckets.
        int _frames, _framesOverBudget, _hitches;
        double _frameSum, _frameMax;
        string _worstFrameStage = "";
        string _currentStage = "";

        public RefineProfile()
        {
            _startManaged = GC.GetTotalMemory(false);
            _startNative = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong();
        }

        public double ElapsedMs => _clock.Elapsed.TotalMilliseconds;

        /// <summary>Wall-time scope for one pipeline stage. Dispose ends it.</summary>
        public StageScope Stage(string name) => new StageScope(this, name);

        public readonly struct StageScope : IDisposable
        {
            readonly RefineProfile _p;
            readonly string _name;
            readonly double _t0;
            readonly string _outer;

            public StageScope(RefineProfile p, string name)
            {
                _p = p; _name = name; _t0 = p.ElapsedMs;
                _outer = p._currentStage;
                p._currentStage = name;
            }

            public void Dispose()
            {
                if (_p == null) return;   // `_profile?.Stage(..) ?? default` when profiling is off
                _p._stages.Add((_name, _p.ElapsedMs - _t0));
                _p._currentStage = _outer;
            }
        }

        /// <summary>Call once per compositor frame the bake was resumed on
        /// (after every <c>await Task.Yield()</c> or readback).</summary>
        public void Frame()
        {
            float dt = Time.unscaledDeltaTime * 1000f;
            _frames++;
            _frameSum += dt;
            if (dt > _frameMax) { _frameMax = dt; _worstFrameStage = _currentStage; }
            if (dt > FrameBudgetMs * 1.05f) _framesOverBudget++;
            if (dt > HitchMs)
            {
                _hitches++;
                Logger.Warning($"[TextureRefine][Hitch] {dt:F0}ms stage={_currentStage}");
            }
        }

        public void Keyframe(double decodeWaitMs, double uploadMs, double refineMs, double issueMs,
            double workerReadMs, double workerDecodeMs, bool decodeFallback)
        {
            _keyframes++;
            _decodeWait += decodeWaitMs; if (decodeWaitMs > _decodeWaitMax) _decodeWaitMax = decodeWaitMs;
            _upload += uploadMs;         if (uploadMs > _uploadMax) _uploadMax = uploadMs;
            _refine += refineMs;         if (refineMs > _refineMax) _refineMax = refineMs;
            _issue += issueMs;           if (issueMs > _issueMax) _issueMax = issueMs;
            _read += workerReadMs;       if (workerReadMs > _readMax) _readMax = workerReadMs;
            _decode += workerDecodeMs;   if (workerDecodeMs > _decodeMax) _decodeMax = workerDecodeMs;
            if (decodeFallback) _decodeFallbacks++;
        }

        public string Report(string title)
        {
            var sb = new StringBuilder(1024);
            sb.Append("[TextureRefine][Profile] ").Append(title)
              .Append(" total=").Append(ElapsedMs.ToString("F0")).Append(" ms");
            sb.Append(" frames=").Append(_frames)
              .Append(" frameMean=").Append((_frames > 0 ? _frameSum / _frames : 0).ToString("F1"))
              .Append(" frameMax=").Append(_frameMax.ToString("F1")).Append('@').Append(_worstFrameStage)
              .Append(" over72Hz=").Append(_framesOverBudget)
              .Append(" hitches>").Append(HitchMs.ToString("F0")).Append("ms=").Append(_hitches);
            long managed = GC.GetTotalMemory(false);
            // Unity's native heap (textures, buffers, meshes); the graphics-driver
            // counter reads 0 on Vulkan Android.
            long native = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong();
            sb.Append(" managedMB=").Append((managed / 1048576.0).ToString("F0"))
              .Append('(').Append(((managed - _startManaged) / 1048576.0).ToString("+0;-0")).Append(')')
              .Append(" nativeMB=").Append((native / 1048576.0).ToString("F0"))
              .Append('(').Append(((native - _startNative) / 1048576.0).ToString("+0;-0")).Append(')');
            sb.Append('\n').Append("  stages:");
            foreach (var (name, ms) in _stages)
                sb.Append(' ').Append(name).Append('=').Append(ms.ToString("F0"));
            if (_keyframes > 0)
            {
                double n = _keyframes;
                sb.Append('\n').Append("  keyframes=").Append(_keyframes)
                  .Append(" main: decodeWait=").Append((_decodeWait / n).ToString("F1")).Append('/').Append(_decodeWaitMax.ToString("F0"))
                  .Append(" upload=").Append((_upload / n).ToString("F1")).Append('/').Append(_uploadMax.ToString("F0"))
                  .Append(" refine=").Append((_refine / n).ToString("F1")).Append('/').Append(_refineMax.ToString("F0"))
                  .Append(" issue=").Append((_issue / n).ToString("F1")).Append('/').Append(_issueMax.ToString("F0"))
                  .Append(" | worker: read=").Append((_read / n).ToString("F1")).Append('/').Append(_readMax.ToString("F0"))
                  .Append(" decode=").Append((_decode / n).ToString("F1")).Append('/').Append(_decodeMax.ToString("F0"))
                  .Append(" fallbacks=").Append(_decodeFallbacks)
                  .Append(" (ms mean/max)");
            }
            return sb.ToString();
        }
    }
}
