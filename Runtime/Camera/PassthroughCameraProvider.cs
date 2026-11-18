using System;
using System.Threading.Tasks;
using Meta.XR;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace Genesis.RoomScan
{
    /// <summary>
    /// Camera provider backed by Meta's PassthroughCameraAccess (Quest 3+).
    /// Provides intrinsics, extrinsics, and RGB frames from the headset cameras.
    ///
    /// <para>
    /// PCA discovery is <b>scene-wide</b> (not GameObject-local): if the Meta XR
    /// Building Block already dropped a <see cref="PassthroughCameraAccess"/>
    /// onto an OVRCameraRig, the provider re-uses it. Without scene-wide find
    /// we ended up with two PCA components fighting over the single native
    /// camera handle — first session worked, subsequent sessions stuck because
    /// PCA self-disables when <c>Play()</c> fails.
    /// </para>
    /// RGB capture starts only from <see cref="StartCapture"/> (scan start)
    /// and stops from <see cref="StopCapture"/>. The scene component is
    /// disabled at Awake so a Building Block left enabled does not open the
    /// headset cameras at boot. HEADSET_CAMERA is still requested by the host
    /// via <see cref="RequestCameraPermissionAsync"/> — that is independent
    /// of starting capture.
    /// </summary>
    [DefaultExecutionOrder(-50)]
    public class PassthroughCameraProvider : MonoBehaviour, ICameraProvider, ICameraFrameTiming
    {
        /// <summary>The Horizon OS permission required by PCA on Quest 3+.</summary>
        public const string CameraPermissionId = AndroidRuntimePermission.Camera;

        [SerializeField] private PassthroughCameraAccess.CameraPositionType cameraPosition =
            PassthroughCameraAccess.CameraPositionType.Left;
        [SerializeField] private Vector2Int requestedResolution = new(1280, 960);
        [SerializeField] private int maxFramerate = 30;

        private PassthroughCameraAccess _pca;

        private void Awake()
        {
            AdoptOrFindPca();
            if (_pca != null)
                _pca.enabled = false;
        }

        /// <inheritdoc />
        public bool IsReady => _pca != null && _pca.IsPlaying && _pca.IsUpdatedThisFrame;

        /// <inheritdoc />
        public bool IsPlaying => _pca != null && _pca.IsPlaying;

        /// <inheritdoc />
        public Texture CurrentFrame => _pca != null && _pca.IsPlaying ? _pca.GetTexture() : null;

        /// <inheritdoc />
        public Pose CameraPose =>
            _pca != null && _pca.IsPlaying ? _pca.GetCameraPose() : Pose.identity;

        /// <inheritdoc />
        public double FrameTimeSeconds =>
            _pca != null && _pca.IsPlaying
                ? (_pca.Timestamp - DateTime.UnixEpoch).TotalSeconds
                : Time.realtimeSinceStartupAsDouble;

        /// <inheritdoc />
        public Vector2 FocalLength =>
            _pca != null && _pca.IsPlaying ? _pca.Intrinsics.FocalLength : Vector2.one;

        /// <inheritdoc />
        public Vector2 PrincipalPoint =>
            _pca != null && _pca.IsPlaying ? _pca.Intrinsics.PrincipalPoint : Vector2.zero;

        /// <inheritdoc />
        public Vector2 SensorResolution =>
            _pca != null && _pca.IsPlaying ? _pca.Intrinsics.SensorResolution : new Vector2(1280, 960);

        /// <inheritdoc />
        public Vector2 CurrentResolution =>
            _pca != null && _pca.IsPlaying
                ? new Vector2(_pca.CurrentResolution.x, _pca.CurrentResolution.y)
                : new Vector2(1280, 960);

        /// <summary>
        /// True when the user has granted the Horizon OS HEADSET_CAMERA
        /// permission. Always true outside Android device builds.
        /// </summary>
        public static bool HasCameraPermission
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            get => Permission.HasUserAuthorizedPermission(CameraPermissionId);
#else
            get => true;
#endif
        }

        /// <summary>
        /// Requests the HEADSET_CAMERA permission and resolves once the user
        /// accepts, denies, or dismisses the system dialog. Always resolves
        /// <c>true</c> outside Android device builds (no permission to request).
        /// Resolves <c>true</c> immediately if already granted.
        /// </summary>
        public static Task<bool> RequestCameraPermissionAsync()
            => AndroidRuntimePermission.RequestAsync(CameraPermissionId);

        /// <inheritdoc />
        public void StartCapture()
        {
            // No permission request here: RoomScanner.StartScanningAsync asks
            // through AndroidRuntimePermission (serialised) before bring-up. A
            // bare RequestUserPermission from this spot raced that queue.
            if (!AndroidRuntimePermission.Has(CameraPermissionId))
                Logger.Warning("HEADSET_CAMERA not granted — PCA will not deliver frames; scanning depth-only.");

            AdoptOrFindPca();
            if (_pca == null) return;

            // PCA forbids MaxFramerate changes while running. Drive it disabled
            // for the property writes, then re-enable so OnEnable runs cleanly.
            // No-op if it was already disabled.
            bool wasEnabled = _pca.enabled;
            if (wasEnabled) _pca.enabled = false;
            _pca.CameraPosition = cameraPosition;
            _pca.RequestedResolution = requestedResolution;
            _pca.MaxFramerate = maxFramerate;
            _pca.enabled = true;
            LogSupportedResolutionsOnce();
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        static bool s_loggedResolutions;
#endif

        /// <summary>The resolutions this headset's camera offers, once per run: which image sizes a scan could ask
        /// for (the texture bake's detail and each keyframe's field of view follow from it).</summary>
        void LogSupportedResolutionsOnce()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (s_loggedResolutions) return;
            s_loggedResolutions = true;
            try
            {
                var list = PassthroughCameraAccess.GetSupportedResolutions(cameraPosition);
                Logger.Info($"PassthroughCameraProvider: requested {requestedResolution.x}x{requestedResolution.y} @ {maxFramerate} fps; " +
                            $"supported: {(list == null ? "none" : string.Join(", ", Array.ConvertAll(list, r => $"{r.x}x{r.y}")))}");
            }
            catch (Exception e)
            {
                Logger.Warning($"PassthroughCameraProvider: could not list supported resolutions: {e.Message}");
            }
#endif
        }

        /// <inheritdoc />
        public void StopCapture()
        {
            if (_pca != null)
                _pca.enabled = false;
        }

        private void AdoptOrFindPca()
        {
            // Scene-wide find: re-use the PCA from the Meta XR Building Block
            // (typically attached to OVRCameraRig). Falling back to a
            // GameObject-local AddComponent here was the bug that caused two
            // PCAs to race for the camera handle — PCA's native side allows
            // exactly one instance per camera position, so the second one
            // self-disables and from then on neither one plays.
            if (_pca != null) return;

            _pca = FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include);
            if (_pca == null)
            {
                Logger.Warning("PassthroughCameraProvider: no PassthroughCameraAccess in scene — adding one to " +
                               $"'{gameObject.name}'. Prefer letting Meta's Building Block place it on the OVRCameraRig.");
                _pca = gameObject.AddComponent<PassthroughCameraAccess>();
                _pca.enabled = false;
            }
            else
            {
                Logger.Info($"PassthroughCameraProvider: adopted existing PassthroughCameraAccess on '{_pca.gameObject.name}'.");
            }
        }

        private void OnDestroy()
        {
            StopCapture();
        }
    }
}
