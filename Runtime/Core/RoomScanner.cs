using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Genesis.RoomScan.UI;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Which representation of the scan to display. <see cref="RoomScanner.CycleRenderMode"/>
    /// automatically skips modes whose backing data or module is not present.
    /// </summary>
    public enum ScanRenderMode
    {
        /// <summary>Live GPU mesh with vertex colors only (triplanar forced off).</summary>
        Vertex,
        /// <summary>Live GPU mesh with triplanar-projected camera textures (falls back to vertex colors where data is missing).</summary>
        Triplanar,
        /// <summary>UV-unwrapped mesh with on-device baked atlas.</summary>
        Refined,
        /// <summary>UV-unwrapped mesh with server-enhanced high-resolution atlas.</summary>
        HQRefined,
        /// <summary>Refined mesh rendered as invisible depth-only occluder for MR.</summary>
        Occlusion,
        /// <summary>Gaussian Splat point cloud rendered from server-trained PLY data.</summary>
        Splat,
        /// <summary>All scan rendering disabled.</summary>
        None,
        /// <summary>Live GPU mesh wireframe via barycentric edge detection.</summary>
        Wireframe
    }

    /// <summary>High-level phase of the scanning session.</summary>
    public enum ScanPhase
    {
        /// <summary>Before <see cref="RoomScanner.StartScanningAsync"/> is called.</summary>
        NotStarted,
        /// <summary>Progress below 0.30 — the mesh is mostly frontier.</summary>
        Discovering,
        /// <summary>Progress 0.30–0.90 — closing holes, surfaces still settling.</summary>
        Refining,
        /// <summary>Progress 0.90–0.95.</summary>
        Stabilized,
        /// <summary>Progress at or above 0.95: closed and refined.</summary>
        Complete
    }

    /// <summary>Raw scan coverage metrics. Updated periodically during scanning.</summary>
    public struct ScanCoverage
    {
        /// <summary>Total depth frames fused so far.</summary>
        public int IntegrationCount;
        /// <summary>Current GPU mesh vertex count.</summary>
        public int MeshVertexCount;
        /// <summary>Current GPU mesh triangle count (indices / 3).</summary>
        public int MeshTriangleCount;
        /// <summary>Voxels near the zero-crossing with sufficient weight.</summary>
        public int SurfaceVoxelCount;
        /// <summary>Frozen surface voxels (user-confirmed done areas).</summary>
        public int FrozenSurfaceCount;
        /// <summary>Surface voxels with camera RGB data.</summary>
        public int ColoredSurfaceCount;
        /// <summary>Colored / Surface ratio (0–1).</summary>
        public float ColorCoverage;
        /// <summary>Frozen / Surface ratio (0–1). The freeze tool's own metric; not part of progress.</summary>
        public float FrozenFraction;
        /// <summary>Number of captured keyframes so far.</summary>
        public int KeyframeCount;

        // ── Analytic (always available; the gate) ─────────────────────
        /// <summary>True once the first analysis cycle of this scan has completed.</summary>
        public bool AnalysisAvailable;
        /// <summary>
        /// Closure 0–1 = surface area / (surface area + leak area). Leak area
        /// is where observed free space meets unknown connected to the outside
        /// of the scan — where passthrough shows through. 1 for a sealed scan.
        /// Faces on a clip plane or the volume edge are cuts, not leaks.
        /// </summary>
        public float Closure;
        /// <summary>
        /// How much of the surface has stopped moving, 0–1: the confident
        /// fraction scaled so the refinement saturation point reads 1.
        /// </summary>
        public float Refinement;
        /// <summary>Confident ÷ surface voxels, raw (0–1).</summary>
        public float ConfidentFraction;
        /// <summary>Surface voxels at or above the confident weight.</summary>
        public int ConfidentSurfaceCount;
        /// <summary>Total leak area, m². The number a host gates on ("under 0.3 m² still open").</summary>
        public float LeakAreaM2;
        /// <summary>Surface area estimate, m².</summary>
        public float SurfaceAreaM2;
        /// <summary>Leak patches at or above the minimum area.</summary>
        public int HoleCount;
        /// <summary>Largest leak patch (the frontier while scanning), or default.</summary>
        public MeshHole LargestHole;
        /// <summary>Leak faces capped by the auto-fill so far this scan.</summary>
        public int LeakFills;

        // ── Shell prior (optional; guidance and auto-fill, never the gate) ──
        /// <summary>
        /// True when the captured room shell (MRUK walls / floor / ceiling /
        /// furniture) is being marched. False without <see cref="RoomUnderstanding"/>
        /// or a resolved room. Guidance only — progress is <see cref="Closure"/>
        /// and <see cref="Refinement"/>.
        /// </summary>
        public bool ShellCoverageAvailable;
        /// <summary>
        /// Fraction of required shell cells with scanned matter on their march
        /// segment (0–1). Door / window openings (build time) and cells whose
        /// segment is observed air (per tick) are not in the denominator.
        /// </summary>
        public float ShellCoverage;
        public int ShellCellsTotal;
        public int ShellCellsCovered;
        public int ShellCellsExcluded;
        /// <summary>Cells the march found to be observed air this tick (nothing there to scan).</summary>
        public int ShellCellsEmpty;
        /// <summary>Connected uncovered patches of two or more cells.</summary>
        public int ShellGapCount;
        /// <summary>Largest uncovered patch, or default when none.</summary>
        public ShellGap LargestGap;
        /// <summary>Auto-fill dispatches so far this scan.</summary>
        public int ShellFillsApplied;
    }

    /// <summary>High-level scan progress combining raw metrics into a single progress value and phase.</summary>
    public struct ScanProgress
    {
        /// <summary>The underlying raw coverage metrics.</summary>
        public ScanCoverage Coverage;
        /// <summary>
        /// <c>Closure × (1 − influence × (1 − Refinement))</c>: the mesh has no
        /// holes, discounted a little while its surface is still moving. 0
        /// until the first analysis cycle. Nothing from the scene model feeds
        /// this.
        /// </summary>
        public float OverallProgress;
        /// <summary>Current high-level scan phase.</summary>
        public ScanPhase Phase;
    }

    /// <summary>
    /// Top-level orchestrator for room scanning. All sibling components live on
    /// the same GameObject and are resolved automatically via GetComponent.
    /// Input bindings are handled by <see cref="RoomScanInputHandler"/> (optional).
    /// </summary>
    [RequireComponent(typeof(DepthCapture), typeof(VolumeIntegrator), typeof(MeshExtractor))]
    [RequireComponent(typeof(RoomScanPersistence), typeof(RoomAnchorManager))]
    public class RoomScanner : MonoBehaviour
    {
        public static RoomScanner Instance { get; private set; }

        [Header("Scan Rates")]
        [SerializeField] private float integrationHz = 30f;
        [SerializeField] private float meshExtractionHz = 8f;

        [Header("Render Mode")]
        [SerializeField] private ScanRenderMode renderMode = ScanRenderMode.Vertex;

        [SerializeField, Range(0.2f, 5f), Tooltip("Wireframe line thickness multiplier")]
        private float wireThickness = 1.5f;

        [SerializeField, Tooltip("Show blue tint overlay on frozen voxels (Vertex/Triplanar/Wireframe modes)")]
        private bool showFreezeTint = true;

        [Header("Logging")]
        [SerializeField] private LogLevel logLevel = LogLevel.Info;

        [Header("Scan Priors")]
        [SerializeField, Tooltip(
            "When true, TSDF stays inside the MRUK room that contained " +
            "the headset at scan start (planes expanded 50 cm outward, " +
            "then hard-confined). Hallway hits past that still skip. " +
            "Default off. Hosts that want a single-room mesh set this " +
            "before StartScanningAsync. No-op without RoomUnderstanding.")]
        private bool confineScanToContainingRoom;

        [SerializeField]
        [Tooltip("Stamp MRUK SCREEN (TV) planes as analytic TSDF. Default on. No-op without RoomUnderstanding.")]
        private bool stampScreenPlanes = true;

        /// <summary>
        /// Opt-in room clip for TSDF. Default false. Set true before
        /// <see cref="StartScanningAsync"/>. No-op without
        /// <see cref="RoomUnderstanding"/>. Changing it during a scan
        /// re-uploads clip planes on the next bind.
        /// </summary>
        public bool ConfineScanToContainingRoom
        {
            get => confineScanToContainingRoom;
            set
            {
                confineScanToContainingRoom = value;
                if (IsScanning) BindScanPriors();
            }
        }

        /// <summary>
        /// When true, MRUK <c>SCREEN</c> (TV) slabs are stamped as analytic
        /// TSDF after each Integrate. Default true. Hosts can turn this
        /// off before <see cref="StartScanningAsync"/> to measure cost or
        /// keep glass depth.
        /// </summary>
        public bool StampScreenPlanes
        {
            get => stampScreenPlanes;
            set
            {
                stampScreenPlanes = value;
                if (IsScanning) BindScanPriors();
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  Sibling component cache (resolved in Awake)
        // ─────────────────────────────────────────────────────────────

        private DepthCapture _depthCapture;
        private VolumeIntegrator _volumeIntegrator;
        private MeshExtractor _meshExtractor;
        private RoomScanPersistence _persistence;
        private RoomAnchorManager _roomAnchor;

        // Optional modules (discovered, not required)
        private PassthroughCameraProvider _cameraProvider;
        private TriplanarCache _triplanarCache;
        private KeyframeCollector _keyframeCollector;
        private IGSplatProvider _gsplatProvider;
        private TextureRefinement _textureRefinement;
        private RoomUnderstanding _roomUnderstanding;
        private DebugMenuController _debugMenu;
        private ICameraProvider _customCameraProvider;
        private IRoomScanModule[] _modules;

        // ─────────────────────────────────────────────────────────────
        //  Public read-only state
        // ─────────────────────────────────────────────────────────────

        /// <summary>Toggle blue tint overlay on frozen voxels in live mesh modes.</summary>
        public bool ShowFreezeTint
        {
            get => showFreezeTint;
            set { showFreezeTint = value; Shader.SetGlobalFloat(NoFreezeTintID, value ? 0f : 1f); }
        }

        /// <summary>
        /// Tint the live mesh red where the surface is open — the boundary
        /// edges the closure metric counts. The same data the analysis uses,
        /// so what is red is what is holding the number down.
        /// </summary>
        public bool ShowHoles
        {
            get => _showHoles;
            set { _showHoles = value; Shader.SetGlobalFloat(ShowHolesID, value ? 1f : 0f); }
        }
        private bool _showHoles;

        /// <summary>The unified scene object registry (MRUK + AI detections).</summary>
        public SceneObjectRegistry SceneObjectRegistry => _sceneObjectRegistry;

        [SerializeField, Tooltip("Show scene object annotations overlay (wireframe boxes + labels)")]
        private bool showSceneObjects;
        [SerializeField, Tooltip("Shader used for debug overlay wireframes and labels")]
        internal Shader debugOverlayShader;

        /// <summary>Toggle scene object annotation overlay on any render mode.</summary>
        public bool ShowSceneObjects
        {
            get => showSceneObjects;
            set
            {
                showSceneObjects = value;
                if (value)
                {
                    if (_sceneObjectVisualizer == null)
                    {
                        var go = new GameObject("SceneObjectVisualizer");
                        go.transform.SetParent(transform, false);
                        _sceneObjectVisualizer = go.AddComponent<SceneObjectVisualizer>();
                        _sceneObjectVisualizer.SetShader(debugOverlayShader);
                    }
                    _sceneObjectVisualizer.Show(_sceneObjectRegistry);
                }
                else
                {
                    _sceneObjectVisualizer?.Hide();
                }
            }
        }

        public bool IsScanning { get; private set; }
        public ScanRenderMode CurrentRenderMode => renderMode;
        public bool IsGsTrainingInProgress => _serverTrainingInProgress;
        public DebugMenuController DebugMenu => _debugMenu;

        /// <summary>The core volume integrator component.</summary>
        public VolumeIntegrator VolumeIntegrator => _volumeIntegrator;
        /// <summary>The core depth capture component.</summary>
        public DepthCapture DepthCapture => _depthCapture;
        /// <summary>The core mesh extractor component.</summary>
        public MeshExtractor MeshExtractor => _meshExtractor;
        /// <summary>The active camera provider (custom or passthrough).</summary>
        public ICameraProvider ActiveCameraProvider => GetActiveCameraProvider();
        /// <summary>The optional Gaussian Splat provider, or null if the GSplat module is not attached.</summary>
        public IGSplatProvider GSplatProvider => _gsplatProvider;
        /// <summary>True when the TextureRefinement module is attached.</summary>
        public bool HasTextureRefinementModule => _textureRefinement != null;

        // ─────────────────────────────────────────────────────────────
        //  Events
        // ─────────────────────────────────────────────────────────────

        public event Action ScanStarted;
        public event Action ScanStopped;
        public event Action<ScanRenderMode> RenderModeChanged;
        /// <summary>Raised when a refined mesh + atlas becomes available (after refinement or package load).</summary>
        public event Action<Mesh, Texture2D> RefinedMeshReady;

        /// <summary>
        /// On-device texture refinement threw (its message). <see cref="RefinedMeshReady"/> will not fire for this run, so
        /// anything waiting on it can stop waiting.
        /// </summary>
        public event Action<string> RefinementFailed;

        /// <summary>
        /// Raised each frame a passthrough camera frame is fed to the volume integrator.
        /// Parameters: frame texture, camera pose, focal length, principal point, sensor resolution, current resolution.
        /// </summary>
        public event Action<Texture, Pose, Vector2, Vector2, Vector2, Vector2> ColorFrameProvided;

        /// <summary>Raised after each depth integration pass.</summary>
        public event Action Integrated;
        /// <summary>Raised after each mesh extraction pass.</summary>
        public event Action MeshExtracted;

        // ─────────────────────────────────────────────────────────────
        //  Private state
        // ─────────────────────────────────────────────────────────────

        private float _lastIntegrationTime;
        private float _lastMeshTime;
        private bool _started;
        private bool _serverTrainingInProgress;
        private bool _scanResourcesReleased;

        // ─────────────────────────────────────────────────────────────
        //  Texture refinement state
        // ─────────────────────────────────────────────────────────────

        private MeshFilter _refinedMeshFilter;
        private MeshRenderer _refinedRenderer;
        private Material _refinedMaterial;
        private Material _occlusionMaterial;
        private Texture2D _refinedAtlasTexture;
        private Texture2D _hqAtlasTexture;
        private Texture2D _normalMapTexture;
        private Mesh _refinedMesh;
        private UnwrappedMeshResult? _cachedUnwrap;

        // Scene object registry
        private SceneObjectRegistry _sceneObjectRegistry;
        private SceneObjectVisualizer _sceneObjectVisualizer;

        public bool HasRefinedTexture { get; private set; }
        public bool HasHQRefinedTexture { get; private set; }

        /// <summary>
        /// True (default): when on-device refinement finishes, switch to
        /// the refined mesh and <see cref="RoomScanSession.FinalizeScanAsync"/>
        /// releases the live TSDF. False: bake into memory
        /// (<see cref="HasRefinedTexture"/>, <see cref="RefinedMeshReady"/>)
        /// but keep drawing the live vertex mesh until the host calls
        /// <see cref="SetRenderMode"/>(<see cref="ScanRenderMode.Refined"/>)
        /// and then <see cref="ReleaseScanResources"/>.
        /// </summary>
        public bool PresentRefinedWhenReady { get; set; } = true;

        /// <summary>The refined-mesh renderer, or null until a refined mesh exists.</summary>
        public MeshRenderer RefinedMeshRenderer => _refinedRenderer;
        public bool IsRefining { get; private set; }
        public bool IsHQRefining { get; private set; }
        public bool IsMeshEnhancing { get; private set; }
        public bool HasEnhancedMesh { get; internal set; }
        public string RefineStatus { get; private set; }
        public string HQRefineStatus { get; private set; }
        public string MeshEnhanceStatus { get; private set; }

        /// <summary>Current scan coverage metrics (updated periodically during scanning).</summary>
        public ScanCoverage CurrentCoverage => BuildCoverage();
        /// <summary>High-level scan progress with blended overall progress and phase.</summary>
        public ScanProgress CurrentProgress => BuildProgress();

        /// <summary>The UV-unwrapped refined mesh (null until refinement completes or a refined package is loaded).</summary>
        public Mesh RefinedMesh => _refinedMesh;
        /// <summary>The on-device baked texture atlas (null until refinement completes or a refined package is loaded).</summary>
        public Texture2D RefinedAtlas => _refinedAtlasTexture;
        /// <summary>The server-enhanced HQ atlas, or null if not available.</summary>
        public Texture2D HQAtlas => _hqAtlasTexture;

        /// <summary>
        /// The original (unsimplified) refined mesh data. Always preserved across re-refines.
        /// Used as the source-of-truth for re-refinement and UV reconstruction.
        /// </summary>
        internal RefinedTextureResult? LastRefinedResult { get; set; }

        /// <summary>
        /// The post-bake simplified mesh (if simplification was applied). Null when
        /// postBakeSimplificationRatio >= 1 or simplification hasn't run.
        /// This is what gets rendered and saved, but re-refinement always uses LastRefinedResult.
        /// </summary>
        internal RefinedTextureResult? LastSimplifiedResult { get; set; }

        /// <summary>Whether a simplified version of the refined mesh exists.</summary>
        public bool HasSimplifiedMesh => LastSimplifiedResult.HasValue;

        /// <summary>
        /// Relocation matrix from the last scan load. Transforms old-session world-space
        /// poses to current world-space. Identity when no relocation occurred or when
        /// running a live (non-reloaded) scan.
        /// </summary>
        internal Matrix4x4 KeyframeRelocation { get; set; } = Matrix4x4.identity;

        private float IntegrationInterval => 1f / integrationHz;
        private float MeshInterval => 1f / meshExtractionHz;

        // ─────────────────────────────────────────────────────────────
        //  Lifecycle
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            Instance = this;
            Logger.Level = logLevel;
            CacheComponents();
            SetSafeShaderDefaults();
        }

        private void Start()
        {
            // Match DepthCapture's Editor early-out: when no XR loader is
            // active, AR/MRUK subsystems are dead and any module that touches
            // them on init will NRE. The user can still load saved scans
            // through other code paths but cannot start a live scan.
            if (!XRRuntimeGuard.IsXRActive)
            {
                Logger.Warning("RoomScanner: " + XRRuntimeGuard.EditorDisabledMessage);
                enabled = false;
                return;
            }

            _modules = GetComponents<IRoomScanModule>();
            foreach (var m in _modules) m.OnModuleInitialize(this);

            if (_persistence != null)
                _persistence.LoadCompleted += ApplyRenderMode;

            SetupHeadExclusion();
            _shellTracker = new ShellCoverageTracker(_shellCells);
            if (_volumeIntegrator != null)
                _volumeIntegrator.ShellResultReady += OnShellResult;

            if (_roomAnchor != null && _roomAnchor.enabled)
            {
                if (_roomAnchor.IsRoomLoaded)
                    CompleteRoomStartup();
                else
                    _roomAnchor.RoomReady += OnRoomAnchorReady;
            }
            else
                CompleteRoomStartup();
        }

        private void OnRoomAnchorReady()
        {
            if (_roomAnchor != null)
                _roomAnchor.RoomReady -= OnRoomAnchorReady;
            if (_started)
                return;
            CompleteRoomStartup();
        }

        private void CacheComponents()
        {
            _depthCapture = GetComponent<DepthCapture>();
            _volumeIntegrator = GetComponent<VolumeIntegrator>();
            _meshExtractor = GetComponent<MeshExtractor>();
            _cameraProvider = GetComponent<PassthroughCameraProvider>();
            _triplanarCache = GetComponent<TriplanarCache>();
            _persistence = GetComponent<RoomScanPersistence>();
            _keyframeCollector = GetComponent<KeyframeCollector>();
            _gsplatProvider = GetComponent<IGSplatProvider>();
            _textureRefinement = GetComponent<TextureRefinement>();
            _roomUnderstanding = GetComponent<RoomUnderstanding>();
            _debugMenu = GetComponentInChildren<DebugMenuController>();
            _roomAnchor = GetComponent<RoomAnchorManager>();
        }

        /// <summary>
        /// Finishes startup after MRUK room is ready (or immediately if <see cref="RoomAnchorManager"/> is disabled).
        /// </summary>
        private void CompleteRoomStartup()
        {
            if (_started)
                return;
            _started = true;
            Logger.Info("Room ready — call StartScanning() to begin");
        }

        private void OnDisable()
        {
            StopScanning();
            UnsubscribeFromAnchorsChanged();
        }

        private void OnDestroy()
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ShellResultReady -= OnShellResult;
        }

        private float _lastScannerLog;
        private int _integrateCount;
        private bool _subscribedToAnchorsChanged;
        private Guid _scanRoomUuid;
        private float _lastEmptyRoomBindAttempt;
        private readonly List<Vector4> _clipScratch = new(32);
        private readonly List<ScanScreenStamp> _stampScratch = new(4);
        private readonly ShellCellSet _shellCells = new();
        private ShellCoverageTracker _shellTracker;
        private float _lastShellLog;
        private OVRCameraRig _bodyRig;
        private OVRHand _leftOvrHand;
        private OVRHand _rightOvrHand;

        private void Update()
        {
            if (_clearDone)
            {
                _clearDone = false;
                _clearInProgress = false;

                // Re-create GPU mesh pipeline (deferred from ClearAllDataAsync to
                // give the GPU a frame to finish using the old buffers).
                if (_meshExtractor != null && !_meshExtractor.IsInitialized)
                    _meshExtractor.Reinitialize();

                Logger.Info("All scan + export data cleared");
                _clearDoneCallback?.Invoke();
                _clearDoneCallback = null;
            }

            // Enforce triplanar override every frame (not just while scanning)
            // so loaded scans don't flash triplanar before ApplyRenderMode runs.
            if (renderMode == ScanRenderMode.Vertex)
                Shader.SetGlobalFloat(TriAvailableID, 0f);

            if (!IsScanning || !DepthCapture.DepthAvailable) return;

            float t = Time.time;

            if (t - _lastIntegrationTime >= IntegrationInterval)
            {
                _lastIntegrationTime = t;

                RefreshBodyAnchors();
                ProvideColorFrame();
                _volumeIntegrator.Integrate();
                Integrated?.Invoke();
                _integrateCount++;

                MaybeRetryScanRoomBind();

                if (!IsRefining && t - _lastMeshTime >= MeshInterval)
                {
                    if (_meshExtractor != null && _meshExtractor.TryExtract())
                    {
                        _lastMeshTime = t;
                        MeshExtracted?.Invoke();
                    }
                }
            }

            _volumeIntegrator.StepAnalysis();

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Logger.Verbose($"Scanner: integrations={_integrateCount}, " +
                    $"depthAvail={DepthCapture.DepthAvailable}");
                if (_volumeIntegrator.AnalysisAvailable)
                {
                    var cl = _volumeIntegrator.Closure;
                    Logger.Info(
                        $"[RoomScanner] Analysis: progress={_volumeIntegrator.Progress:P0} closure={cl.Closure:P0} " +
                        $"refinement={_volumeIntegrator.Refinement:P0} (confident {_volumeIntegrator.ConfidentFraction:P0}) " +
                        $"leak={cl.LeakAreaM2:F2}m2 surface={cl.SurfaceAreaM2:F1}m2 holes={cl.HoleCount} " +
                        $"largest={cl.LargestHole.AreaM2:F2}m2 (~{cl.LargestHole.ApproxWidthMetres:F2}m across) " +
                        $"@({cl.LargestHole.Center.x:F2},{cl.LargestHole.Center.y:F2},{cl.LargestHole.Center.z:F2}) " +
                        $"faces: leak={cl.LeakFaces} cut={cl.CutFaces} " +
                        $"voxels: surface={_volumeIntegrator.SurfaceVoxelCount} confident={_volumeIntegrator.ConfidentSurfaceCount} " +
                        $"fills: leak={_volumeIntegrator.LeakFills} shell={(_shellTracker != null && _shellTracker.Available ? _shellTracker.FillsApplied : 0)}" +
                        (_shellTracker != null && _shellTracker.Available ? $" shellCov={_shellTracker.Coverage:P0} shellGaps={_shellTracker.GapCount}" : ""));
                }
            }
        }

        // ═════════════════════════════════════════════════════════════
        //  PUBLIC API — call from any client, input handler, or UI
        // ═════════════════════════════════════════════════════════════

        /// <summary>
        /// Begins depth integration and mesh extraction. Resets relocation state,
        /// clears in-memory keyframes, and starts the active camera provider.
        ///
        /// <para>
        /// <b>Permissions are requested here</b>, in sequence, for whatever is
        /// still missing: <c>USE_SCENE</c> (required — a denial aborts the
        /// start), then <c>HEADSET_CAMERA</c> and <c>USE_ANCHOR_API</c> (a
        /// denial degrades). Hosts that want the dialogs at boot for UX call
        /// <see cref="RoomScanSession.RequestScenePermissionAsync"/> and
        /// friends first; the requests here are then no-ops. Nothing else in
        /// the package calls <c>RequestUserPermission</c>.
        /// </para>
        ///
        /// <para>
        /// <b>Async by necessity, not by API preference.</b> The lazy GPU
        /// bring-up (~150 MB TSDF + ~480 MB Surface Nets) is staged across
        /// frames: allocate the volumes, yield twice so the render thread
        /// commits them and runs the first Clear, allocate the mesh
        /// extractor, yield twice again, then switch to the live preview
        /// and enable PCA + AROcclusionManager. Total wall-clock on a
        /// Quest 3 is ~56 ms (4 frames at 72 fps), below the threshold at
        /// which a press feels unregistered. Allocating 600 MB and opening
        /// two camera pipelines in a single frame is a worst case for the
        /// render thread that the staging avoids.
        /// </para>
        ///
        /// <para>
        /// History: a multi-second "hang on the first A-press" was once
        /// attributed to this timing (PCA's MRUK handshake losing a race
        /// to the compute dispatches). The measured cause was different —
        /// <see cref="GPUSurfaceNets"/> drew from an indirect-args buffer
        /// that nothing had written yet, and whether that garbage happened
        /// to be zero depended on what the host had just freed. Eager
        /// allocation at boot avoided it only because boot memory was
        /// clean. The buffer is now zeroed at allocation; the staging is
        /// kept because it is cheap and spreads the render-thread load.
        /// </para>
        /// </summary>
        public async Task StartScanningAsync()
        {
            if (IsScanning || _startingScan) return;

            // Permissions first, before anything is torn down or allocated:
            // the user may take a while on the dialogs, and a denied USE_SCENE
            // means there is no scan to start. Requests are serialised inside
            // AndroidRuntimePermission and free once granted, so a host that
            // already asked at boot pays nothing here.
            _startingScan = true;
            try
            {
                if (!await EnsureScanPermissionsAsync())
                    return;
                await StartScanningCoreAsync();
            }
            finally
            {
                _startingScan = false;
            }
        }

        private bool _startingScan;

        /// <summary>
        /// USE_SCENE is required (no depth, no scan). HEADSET_CAMERA and
        /// USE_ANCHOR_API are requested too but a denial only degrades:
        /// depth-only colour from normals, and a save without a spatial
        /// anchor (floor-anchor relocation fallback).
        /// </summary>
        private async Task<bool> EnsureScanPermissionsAsync()
        {
            if (!await AndroidRuntimePermission.RequestAsync(AndroidRuntimePermission.Scene))
            {
                Logger.Error("USE_SCENE denied — the depth sensor is required to scan. Not starting.");
                return false;
            }
            if (!await AndroidRuntimePermission.RequestAsync(AndroidRuntimePermission.Camera))
                Logger.Warning("HEADSET_CAMERA denied — scanning depth-only; vertex colour falls back to normals.");
            if (!await AndroidRuntimePermission.RequestAsync(AndroidRuntimePermission.Anchors))
                Logger.Warning("USE_ANCHOR_API denied — this scan will save without a spatial anchor.");
            return true;
        }

        private async Task StartScanningCoreAsync()
        {
            // Resume is "same-session pause of an in-progress _tmp scan".
            // Compute it before UnloadActiveScan, which zeros IntegrationCount.
            bool resuming = _persistence != null
                && _persistence.ActivePackageId == RoomScanPersistence.TmpPkgId
                && _volumeIntegrator != null
                && _volumeIntegrator.IntegrationCount > 0;

            // A LoadAsync in this session leaves the refined mesh drawing and
            // a spatial anchor localized. Starting a new scan on top of that
            // keeps the old mesh on screen and would create a second spatial
            // anchor while the first is still bound. Drop the in-memory scan
            // first; saved packages stay on disk.
            if (!resuming)
            {
                UnloadActiveScan();
                await Task.Yield();
                await Task.Yield();
            }

            IsScanning = true;
            try
            {
                KeyframeRelocation = Matrix4x4.identity;
                _cachedUnwrap = null;

                // ── Stage 1: GPU volume bring-up ────────────────────────
                // Both calls are idempotent: ReallocateVolumes early-returns
                // if RTs already exist; EnsureInitialized / Reinitialize
                // early-return for the same reason on the mesh side. The
                // _scanResourcesReleased branch uses Reinitialize because
                // ReleaseScanResources explicitly disposes the mesh
                // extractor and we need a true rebuild, not a no-op.
                _volumeIntegrator.ReallocateVolumes();

                // Yield twice so the render thread can (a) actually commit
                // the two 256³ 3D RT allocations to VRAM, and (b) run the
                // first Clear compute dispatch. One yield is "next frame";
                // two yields gives a clean frame in between before the
                // much bigger Surface Nets alloc lands.
                await Task.Yield();
                await Task.Yield();

                // ── Stage 2: Surface Nets mesh extractor ────────────────
                if (_scanResourcesReleased)
                {
                    _meshExtractor.Reinitialize();
                    _scanResourcesReleased = false;
                }
                else
                {
                    _meshExtractor.EnsureInitialized();
                }

                // Yield twice so the render thread can commit the ~480 MB
                // ComputeBuffers + cell-table upload before PCA enables.
                // This is the critical pair — without it, PCA's native
                // OnEnable lands in the same frame as the first Surface
                // Nets dispatch and the MRUK fence handshake fails.
                await Task.Yield();
                await Task.Yield();

                // ── Stage 3: persistence + live preview ─────────────────
                // In-memory load state was dropped in UnloadActiveScan above
                // when !resuming. GPU is up now, so switch to the live
                // vertex preview and open a fresh _tmp package + anchor.
                if (!resuming)
                {
                    _volumeIntegrator.ResetAnalysis();

                    SetRenderMode(ScanRenderMode.Vertex);

                    if (_keyframeCollector != null)
                        _keyframeCollector.ClearInMemory();

                    _persistence?.CreateTmpPackage();
                    _ = CreateScanAnchorAsync();
                }

                _keyframeCollector?.SetExportDirectory(
                    _persistence != null && _persistence.HasActivePackage
                        ? Path.Combine(_persistence.ActivePackageDirectory, "keyframes")
                        : null);

                float t = Time.time;
                _lastIntegrationTime = t;
                _lastMeshTime = t;

                _cameraAvailable = false;

                // ── Stage 4: camera + depth (now safe) ──────────────────
                // PCA's native OnEnable handshakes with MRUK to grab the
                // passthrough hardware buffer queue. By this point the
                // GPU resources are committed and the render-thread queue
                // is clean, so PCA can win the handshake and MRUK keeps
                // pulling frames steadily.
                ICameraProvider provider = GetActiveCameraProvider();
                provider?.StartCapture();
                _depthCapture.StartDepthCapture();

                if (!resuming)
                {
                    // Fresh scan: reset registry — stale AI detections from a previous
                    // session/load are in a different anchor frame and must be discarded.
                    _sceneObjectRegistry = new SceneObjectRegistry();
                }
                else
                {
                    _sceneObjectRegistry ??= new SceneObjectRegistry();
                }
                PopulateSceneObjectRegistry();
                SubscribeToAnchorsChanged();
                ResolveScanRoomUuid();
                BindScanPriors();

                Logger.Info($"StartScanning — resuming={resuming}, integrationCount={_volumeIntegrator.IntegrationCount}");
                ScanStarted?.Invoke();
                if (_modules != null)
                    foreach (var m in _modules) m.OnScanStarted();
            }
            catch
            {
                // Reset the re-entry guard so the user can retry. Without
                // this, a throw mid-warmup would leave IsScanning=true
                // forever and every subsequent A-press would no-op.
                IsScanning = false;
                throw;
            }
        }

        /// <summary>
        /// Creates a spatial anchor at scan start for persistence. The anchor's matrix
        /// serves as the baseMatrixAtSave for delta relocation on reload — same pattern
        /// used by TSDF, refined mesh, and splat. Placed at the camera's current position
        /// with identity rotation to avoid inheriting MRUK floor's arbitrary yaw.
        /// </summary>
        private async Task CreateScanAnchorAsync()
        {
            var mgr = RoomAnchorManager.Instance;
            if (mgr == null || !mgr.IsRoomLoaded) return;

            var camPos = Camera.main != null ? Camera.main.transform.position : Vector3.zero;
            var result = await mgr.CreateAndSaveSpatialAnchorAsync(camPos, Quaternion.identity);
            if (result.HasValue && _persistence != null && _persistence.HasActivePackage)
            {
                _persistence.WriteSessionAnchorData(
                    result.Value.uuid.ToString(), result.Value.matrix);
            }
        }

        /// <summary>
        /// Pauses depth integration and stops the camera provider.
        /// </summary>
        public void StopScanning()
        {
            if (!IsScanning) return;
            IsScanning = false;

            ICameraProvider provider = GetActiveCameraProvider();
            provider?.StopCapture();
            _depthCapture.StopDepthCapture();

            _volumeIntegrator?.ClearScanPriors();
            _volumeIntegrator?.ClearShellCells();
            _shellTracker?.Disable();
            _scanRoomUuid = Guid.Empty;
            _keyframeCollector?.LogProfile();

            ScanStopped?.Invoke();
            if (_modules != null)
                foreach (var m in _modules) m.OnScanStopped();
        }

        /// <summary>
        /// Toggles between <see cref="StartScanningAsync"/> and <see cref="StopScanning"/>.
        /// The Start path is fire-and-forget here because this is a debug/dev API
        /// (typically wired to a debug-menu button) and the small ~56 ms delay
        /// before integration begins is not worth changing the toggle's signature
        /// for. Production callers should await <see cref="StartScanningAsync"/>
        /// directly so they can sequence UI feedback around the start.
        /// </summary>
        public void ToggleScanning()
        {
            if (IsScanning) StopScanning();
            else _ = StartScanningAsync();
        }

        /// <summary>True after <see cref="ReleaseScanResources"/> has been called. Cleared when scanning restarts.</summary>
        public bool ScanResourcesReleased => _scanResourcesReleased;

        /// <summary>
        /// Frees heavy GPU resources (TSDF volumes, Surface Nets buffers, depth textures) to reclaim
        /// ~400-500 MB of GPU memory. Call after scanning + refinement is complete, before entering gameplay.
        /// The refined MeshRenderer stays alive for game-phase rendering.
        /// Call <see cref="StartScanningAsync"/> to re-allocate everything if a new scan is needed.
        /// </summary>
        public void ReleaseScanResources()
        {
            if (_scanResourcesReleased) return;
            StopScanning();

            _volumeIntegrator.ReleaseVolumes();
            _meshExtractor.DisposeOnly();
            _depthCapture.ReleaseResources();

            if (_triplanarCache != null) _triplanarCache.Clear();

            _scanResourcesReleased = true;

            if (renderMode is ScanRenderMode.Vertex or ScanRenderMode.Wireframe or ScanRenderMode.Triplanar)
                SetRenderMode(HasRefinedTexture ? ScanRenderMode.Refined : ScanRenderMode.None);

            Logger.Info("Scan GPU resources released (~400-500 MB freed)");
        }

        /// <summary>
        /// Drops the in-memory loaded / refined scan so the next
        /// <see cref="StartScanningAsync"/> is an empty-room start. Does
        /// <b>not</b> delete saved packages or erase spatial-anchor UUIDs
        /// from Horizon OS — use <see cref="RoomScanPersistence.DeletePackageAsync"/>
        /// for that.
        /// <para>
        /// Hides the refined mesh, unbinds the active spatial anchor
        /// (children detached first), clears the active-package pointer,
        /// cleans <c>_tmp</c>, and disposes scan GPU resources if they
        /// were allocated. Idempotent.
        /// </para>
        /// </summary>
        public void UnloadActiveScan()
        {
            StopScanning();

            HasRefinedTexture = false;
            HasHQRefinedTexture = false;
            HasEnhancedMesh = false;
            LastRefinedResult = null;
            LastSimplifiedResult = null;
            _cachedUnwrap = null;
            if (_refinedMeshFilter != null)
                _refinedMeshFilter.sharedMesh = null;
            if (_refinedMesh != null)
            {
                Destroy(_refinedMesh);
                _refinedMesh = null;
            }
            if (_normalMapTexture != null)
            {
                Destroy(_normalMapTexture);
                _normalMapTexture = null;
            }
            _gsplatProvider?.ClearSplat();
            _gsplatProvider?.ResetSplatTransform();
            _downloadedPlyData = null;

            SetRenderMode(ScanRenderMode.None);

            RoomAnchorManager.Instance?.UnloadActiveSpatialAnchor();

            if (_keyframeCollector != null)
                _keyframeCollector.ClearInMemory();

            if (_persistence != null)
            {
                _persistence.CleanupTmpPackage();
                _persistence.ClearActivePackage();
            }

            if (!_scanResourcesReleased)
            {
                _volumeIntegrator?.ReleaseVolumes();
                _meshExtractor?.DisposeOnly();
                _depthCapture?.ReleaseResources();
                if (_triplanarCache != null)
                    _triplanarCache.Clear();
                _scanResourcesReleased = true;
            }
            else
            {
                _volumeIntegrator?.Clear();
                if (_triplanarCache != null)
                    _triplanarCache.Clear();
            }

            Logger.Info("Active scan unloaded (saved packages kept)");
        }

        /// <summary>
        /// Clears the TSDF volume and reinitializes the GPU mesh pipeline.
        /// Does not delete persisted files — use <see cref="ClearAllDataAsync"/> for a full wipe.
        /// </summary>
        public void ClearScan()
        {
            _volumeIntegrator.Clear();
            _meshExtractor.Reinitialize();
            if (_triplanarCache != null)
                _triplanarCache.Clear();
        }

        /// <summary>
        /// Clears all persisted data: in-memory scan, saved scan files, triplanar
        /// textures, and temporary package. Safe to call at runtime.
        /// File I/O runs on a background thread via ThreadPool to avoid main-thread
        /// stalls and potential SynchronizationContext deadlocks on Quest/IL2CPP.
        /// GPU resources are disposed without immediate re-allocation to avoid
        /// Vulkan stalls when the GPU is still referencing the previous frame's buffers.
        /// Re-initialization happens lazily on the next <see cref="StartScanningAsync"/> or load.
        /// </summary>
        public void ClearAllDataAsync(Action onComplete = null)
        {
            if (_clearInProgress) return;
            _clearInProgress = true;

            try
            {
                StopScanning();

                _gsplatProvider?.ClearSplat();
                _gsplatProvider?.ResetSplatTransform();
                _downloadedPlyData = null;
                HasRefinedTexture = false;
                HasHQRefinedTexture = false;
                HasEnhancedMesh = false;
                LastRefinedResult = null;
                LastSimplifiedResult = null;
                _cachedUnwrap = null;
                _refinedMesh = null;
                if (_normalMapTexture != null)
                {
                    Destroy(_normalMapTexture);
                    _normalMapTexture = null;
                }

                _meshExtractor.DisposeOnly();
                _volumeIntegrator.Clear();
                if (_triplanarCache != null)
                    _triplanarCache.Clear();

                if (_keyframeCollector != null)
                    _keyframeCollector.ClearInMemory();

                if (_persistence != null)
                {
                    _persistence.CleanupTmpPackage();
                    _persistence.ClearActivePackage();
                }
            }
            catch (Exception e)
            {
                Logger.Error($"ClearAllData sync error: {e.Message}\n{e.StackTrace}");
                _clearInProgress = false;
                return;
            }

            _clearDoneCallback = onComplete;
            _clearDone = true;
        }

        private volatile bool _clearInProgress;
        private volatile bool _clearDone;
        private Action _clearDoneCallback;

        /// <summary>
        /// Switches the active render mode and updates mesh/splat visibility accordingly.
        /// </summary>
        public void SetRenderMode(ScanRenderMode newMode)
        {
            renderMode = newMode;
            ApplyRenderMode();
            RenderModeChanged?.Invoke(renderMode);
            Logger.Info($"Render mode: {renderMode}");
        }

        /// <summary>
        /// Advances to the next available render mode, skipping modes whose backing
        /// data or module is not present.
        /// </summary>
        public void CycleRenderMode()
        {
            ScanRenderMode[] order =
            {
                ScanRenderMode.Wireframe, ScanRenderMode.Vertex, ScanRenderMode.Triplanar,
                ScanRenderMode.Refined, ScanRenderMode.HQRefined, ScanRenderMode.Occlusion,
                ScanRenderMode.Splat, ScanRenderMode.None
            };

            int cur = Array.IndexOf(order, renderMode);
            if (cur < 0) cur = 0;

            for (int i = 1; i <= order.Length; i++)
            {
                var candidate = order[(cur + i) % order.Length];
                if (!IsModeAvailable(candidate)) continue;

                if (candidate == ScanRenderMode.Splat && HasDownloadedSplat
                    && (_gsplatProvider == null || !_gsplatProvider.HasServerTrainedSplats))
                {
                    LoadDownloadedSplat();
                    return;
                }

                SetRenderMode(candidate);
                return;
            }
        }

        /// <summary>
        /// Returns true if the given render mode's backing module/data is present.
        /// </summary>
        public bool IsModeAvailable(ScanRenderMode mode)
        {
            return mode switch
            {
                ScanRenderMode.Vertex => !_scanResourcesReleased,
                ScanRenderMode.Wireframe => !_scanResourcesReleased,
                ScanRenderMode.None => true,
                ScanRenderMode.Triplanar => !_scanResourcesReleased && _triplanarCache != null,
                ScanRenderMode.Refined => HasRefinedTexture,
                ScanRenderMode.HQRefined => HasHQRefinedTexture,
                ScanRenderMode.Occlusion => HasRefinedTexture && _occlusionMaterial != null,
                ScanRenderMode.Splat => (_gsplatProvider != null && _gsplatProvider.HasServerTrainedSplats) || HasDownloadedSplat,
                _ => false
            };
        }

        /// <summary>
        /// Freezes voxels inside the default spotlight cone in front of the
        /// head (see <see cref="FreezeConeHalfAngle"/>). Uses the head pose,
        /// which is always available — not the passthrough camera, whose
        /// intrinsics were not. Hosts that have their own emitter can call
        /// the overload with origin, axis, half-angle, and length.
        /// </summary>
        public void FreezeInView()
        {
            if (_volumeIntegrator == null) return;
            if (!TryGetGaze(out var eye, out var gaze)) return;

            RefreshBodyAnchors();
            _volumeIntegrator.FreezeInView(eye, gaze);
        }

        /// <summary>
        /// Freeze voxels inside a host-supplied spotlight cone instead of the
        /// headset gaze. <paramref name="maxMetres"/> 0 is unbounded.
        /// </summary>
        public void FreezeInView(Vector3 origin, Vector3 direction, float halfAngleDegrees, float maxMetres = 0f)
        {
            if (_volumeIntegrator == null) return;
            RefreshBodyAnchors();
            _volumeIntegrator.FreezeInView(origin, direction, halfAngleDegrees, maxMetres);
        }

        /// <summary>Unfreezes frozen voxels inside the default head cone.</summary>
        public void UnfreezeInView()
        {
            if (_volumeIntegrator == null) return;
            if (!TryGetGaze(out var eye, out var gaze)) return;

            _volumeIntegrator.UnfreezeInView(eye, gaze);
        }

        /// <summary>Unfreeze frozen voxels inside a host-supplied spotlight cone.</summary>
        public void UnfreezeInView(Vector3 origin, Vector3 direction, float halfAngleDegrees, float maxMetres = 0f)
        {
            if (_volumeIntegrator == null) return;
            _volumeIntegrator.UnfreezeInView(origin, direction, halfAngleDegrees, maxMetres);
        }

        /// <summary>Half-angle, degrees, of the default head freeze cone.</summary>
        public float FreezeConeHalfAngle => _volumeIntegrator != null ? _volumeIntegrator.FreezeConeHalfAngle : 15f;

        bool TryGetGaze(out Vector3 eye, out Vector3 gaze)
        {
            RefreshBodyAnchors();
            var head = _volumeIntegrator != null ? _volumeIntegrator.HeadAnchor : null;
            if (head == null && Camera.main != null) head = Camera.main.transform;
            if (head == null)
            {
                eye = default;
                gaze = default;
                return false;
            }
            eye = head.position;
            gaze = head.forward;
            return true;
        }

        /// <summary>
        /// Persists the current scan (TSDF volume, keyframes, artifacts) to a new package on disk.
        /// Returns false if scanning is active — stop scanning before saving.
        /// </summary>
        public async Task<bool> SaveScanAsync()
        {
            if (_persistence == null) return false;
            if (IsScanning)
            {
                Logger.Warning("Cannot save while scanning — stop scan first");
                return false;
            }
            bool ok = await _persistence.SaveToNewPackageAsync();
            if (ok && _keyframeCollector != null)
                _keyframeCollector.SetExportDirectory(KeyframeDirectory);
            if (ok && _sceneObjectRegistry != null && _sceneObjectRegistry.Count > 0)
                await _persistence.SaveSceneObjectsAsync(_sceneObjectRegistry);
            return ok;
        }

        /// <summary>
        /// Loads a previously saved scan package by ID, restoring volume data and artifacts.
        /// </summary>
        public async Task<bool> LoadPackageAsync(string pkgId)
        {
            if (_persistence == null) return false;
            return await _persistence.LoadPackageAsync(pkgId);
        }

        /// <summary>
        /// Lightweight load path: reads only the refined mesh + atlas from a saved package,
        /// skipping TSDF reconstruction, Surface Nets, and triplanar data. Ideal for game-mode
        /// sessions where only the baked mesh is needed. Typically completes in under 1 second.
        /// </summary>
        public async Task<bool> LoadRefinedOnlyAsync(string pkgId)
        {
            if (_persistence == null) return false;
            return await _persistence.LoadRefinedOnlyAsync(pkgId);
        }

        /// <summary>
        /// Exports the current GPU mesh as a PLY point cloud to the active keyframe directory.
        /// </summary>
        public Task ExportPointCloudAsync() => PointCloudExporter.ExportAsync(KeyframeDirectory);

        /// <summary>
        /// Kicks off the server-side Gaussian Splat training pipeline. Downloads the trained PLY on completion.
        /// </summary>
        public void StartServerTraining()
        {
            if (_serverTrainingInProgress) return;
            RunServerTrainingAsync();
        }

        /// <summary>Shows or hides the debug menu HUD if present.</summary>
        public void ToggleDebugMenu()
        {
            if (_debugMenu != null) _debugMenu.Toggle();
        }

        /// <summary>
        /// Set a custom camera provider (overrides PassthroughCameraProvider).
        /// </summary>
        public void SetCameraProvider(ICameraProvider provider)
        {
            _customCameraProvider = provider;
        }

        /// <summary>
        /// Registers an extra torso exclusion capsule at <paramref name="t"/>.
        /// Head and hands are <see cref="SetBodyExclusionAnchors"/>, not this list.
        /// </summary>
        public void AddExclusionZone(Transform t)
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ExclusionZones.Add(t);
        }

        /// <summary>
        /// Unregisters a previously added extra exclusion zone.
        /// </summary>
        public void RemoveExclusionZone(Transform t)
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ExclusionZones.Remove(t);
        }

        /// <summary>
        /// Pin head and wrist transforms for body-exclusion capsules.
        /// Host-owned: the scanner will not overwrite them from the camera rig.
        /// Call before <see cref="StartScanningAsync"/>.
        /// </summary>
        public void SetBodyExclusionAnchors(Transform head, Transform leftHand, Transform rightHand)
        {
            _volumeIntegrator?.SetBodyExclusionAnchors(head, leftHand, rightHand, hostOwned: true);
        }

        // ─────────────────────────────────────────────────────────────
        //  Texture Refinement
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Runs the on-device texture refinement pipeline: UV-unwraps the mesh via xatlas,
        /// bakes a texture atlas from collected keyframes, and switches to Refined render mode.
        /// </summary>
        public async void StartTextureRefinement()
        {
            if (IsRefining) return;
            if (_textureRefinement == null)
            {
                Logger.Warning("TextureRefinement module not attached — skipping refinement");
                return;
            }
            IsRefining = true;
            RefineStatus = "Starting...";

            Action<string> statusHandler = s => RefineStatus = s;
            _textureRefinement.StatusChanged += statusHandler;
            try
            {
                // Freeze the look: no more integrates / keyframes, and no
                // morph-held dump as the unwrap source.
                if (IsScanning)
                    StopScanning();
                string keyframeDir = KeyframeDirectory;
                var unwrap = await EnsureUnwrappedAsync();
                var (atlasPixels, normalPixels) = await _textureRefinement.BakeAtlasAsync(
                    unwrap, keyframeDir, KeyframeRelocation);

                var original = new RefinedTextureResult
                {
                    Positions = unwrap.Positions,
                    Normals = unwrap.Normals,
                    UVs = unwrap.UVs,
                    Indices = unwrap.Indices,
                    AtlasPixels = atlasPixels,
                    NormalPixels = normalPixels,
                    AtlasWidth = unwrap.AtlasWidth,
                    AtlasHeight = unwrap.AtlasHeight
                };

                LastRefinedResult = original;
                LastSimplifiedResult = null;

                var toRender = original;
                if (_textureRefinement.postBakeSimplificationRatio < 1f && !_textureRefinement.simplifyBeforeUnwrap)
                {
                    var simplified = await _textureRefinement.SimplifyRefinedMeshAsync(original);
                    LastSimplifiedResult = simplified;
                    toRender = simplified;
                }

                await ApplyRefinedAtlasAsync(toRender);
                HasRefinedTexture = true;
                if (PresentRefinedWhenReady)
                    SetRenderMode(ScanRenderMode.Refined);

                // IMPORTANT: persist refined artifacts BEFORE firing RefinedMeshReady.
                // RoomScanSession.FinalizeScanAsync wakes on this event and then calls
                // SaveScanAsync → Directory.Move(_tmp → pkg_xxx). If the event fired first,
                // SaveArtifactAsync would race against that Move (the captured pkgDir would
                // point at _tmp while the directory was being renamed), and refined_mesh.bin
                // would silently fail to land in the final package.
                if (_persistence != null && _persistence.HasActivePackage)
                {
                    await _persistence.SaveArtifactAsync(ArtifactType.Refined, null, original);
                    if (LastSimplifiedResult.HasValue)
                        await _persistence.SaveArtifactAsync(ArtifactType.SimplifiedMesh, null, LastSimplifiedResult);
                }

                RefinedMeshReady?.Invoke(_refinedMesh, _refinedAtlasTexture);

                Logger.Info("On-device texture refinement complete");
            }
            catch (Exception e)
            {
                Logger.Error($"Texture refinement failed: {e.Message}\n{e.StackTrace}");
                RefineStatus = "Failed";
                RefinementFailed?.Invoke(e.Message);
            }
            finally
            {
                _textureRefinement.StatusChanged -= statusHandler;
                IsRefining = false;
            }
        }

        /// <summary>
        /// Uploads the on-device atlas to the server for super-resolution enhancement.
        /// Triggers on-device refinement first if it hasn't been run yet.
        /// </summary>
        public async void StartHQRefinement()
        {
            if (IsHQRefining) return;
            if (_textureRefinement == null)
            {
                Logger.Warning("TextureRefinement module not attached — skipping HQ refinement");
                return;
            }
            IsHQRefining = true;
            HQRefineStatus = "Starting...";

            try
            {
                if (_gsplatProvider == null)
                {
                    HQRefineStatus = "No server configured";
                    return;
                }

                // Auto-trigger on-device refinement if not done yet
                if (!LastRefinedResult.HasValue)
                {
                    HQRefineStatus = "Running on-device refine first...";
                    Action<string> hqStatusHandler = s => HQRefineStatus = s;
                    _textureRefinement.StatusChanged += hqStatusHandler;
                    try
                    {
                        StartTextureRefinement();
                        while (IsRefining) await Task.Yield();
                    }
                    finally { _textureRefinement.StatusChanged -= hqStatusHandler; }

                    if (!LastRefinedResult.HasValue)
                    {
                        HQRefineStatus = "On-device refine failed";
                        return;
                    }
                }

                // Encode current refined atlas to PNG
                HQRefineStatus = "Encoding atlas...";
                byte[] pngBytes;
                var r = LastRefinedResult.Value;
                if (r.AtlasPixels != null)
                {
                    var srcTex = new Texture2D(r.AtlasWidth, r.AtlasHeight, TextureFormat.RGBA32, false);
                    srcTex.SetPixelData(r.AtlasPixels, 0);
                    srcTex.Apply();
                    pngBytes = ImageConversion.EncodeToPNG(srcTex);
                    UnityEngine.Object.Destroy(srcTex);
                }
                else if (_refinedAtlasTexture != null)
                {
                    pngBytes = ImageConversion.EncodeToPNG(_refinedAtlasTexture);
                }
                else
                {
                    HQRefineStatus = "No atlas data available";
                    return;
                }

                int scale = _textureRefinement != null ? _textureRefinement.hqRefineScale : 2;
                Logger.Info($"HQ refine: uploading {pngBytes.Length / 1024}KB atlas, scale={scale}");
                HQRefineStatus = $"Uploading ({pngBytes.Length / 1024}KB)...";

                byte[] resultPng = await _gsplatProvider.EnhanceAtlasAsync(pngBytes, scale, inpaint: true);

                if (resultPng == null || resultPng.Length == 0)
                {
                    HQRefineStatus = "Server returned no data";
                    return;
                }

                HQRefineStatus = "Applying atlas...";
                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (ImageConversion.LoadImage(tex, resultPng))
                {
                    _hqAtlasTexture = tex;
                    HasHQRefinedTexture = true;
                    EnsureRefinedRenderer();
                    SetRenderMode(ScanRenderMode.HQRefined);

                    if (_persistence != null && _persistence.HasActivePackage)
                        await _persistence.SaveArtifactAsync(ArtifactType.HQRefined, resultPng);

                    HQRefineStatus = "Done";
                    Logger.Info($"HQ atlas enhancement complete: {tex.width}x{tex.height}");
                }
                else
                {
                    UnityEngine.Object.Destroy(tex);
                    HQRefineStatus = "Failed to decode atlas";
                }
            }
            catch (Exception e)
            {
                Logger.Error($"HQ refinement failed: {e.Message}\n{e.StackTrace}");
                HQRefineStatus = "Failed";
            }
            finally
            {
                IsHQRefining = false;
            }
        }

        /// <summary>
        /// Uploads the refined mesh to the server for smoothing and plane-snapping enhancement.
        /// Requires on-device refinement to have completed first.
        /// </summary>
        public async void StartMeshEnhancement()
        {
            if (IsMeshEnhancing) return;
            if (_textureRefinement == null)
            {
                Logger.Warning("TextureRefinement module not attached — skipping mesh enhancement");
                return;
            }
            IsMeshEnhancing = true;
            MeshEnhanceStatus = "Starting...";

            try
            {
                if (_gsplatProvider == null)
                {
                    MeshEnhanceStatus = "No server configured";
                    return;
                }

                if (!LastRefinedResult.HasValue)
                {
                    MeshEnhanceStatus = "No refined mesh — refine first";
                    return;
                }

                MeshEnhanceStatus = "Serializing mesh...";
                var r = LastRefinedResult.Value;
                byte[] meshBin = await Task.Run(() => SerializeRefinedMesh(r));
                Logger.Info($"Mesh enhance: uploading {meshBin.Length / 1024}KB ({r.Positions.Length} verts)");

                MeshEnhanceStatus = $"Uploading ({meshBin.Length / 1024}KB)...";
                byte[] resultBin = await _gsplatProvider.EnhanceMeshAsync(
                    meshBin, smoothIterations: 3, enablePlaneSnap: false);

                if (resultBin == null || resultBin.Length == 0)
                {
                    MeshEnhanceStatus = "Server returned no data";
                    return;
                }

                MeshEnhanceStatus = "Applying enhanced mesh...";
                var enhanced = await Task.Run(() => DeserializeRefinedMesh(resultBin));
                enhanced.AtlasPixels = r.AtlasPixels;

                ApplyEnhancedMesh(enhanced);
                HasEnhancedMesh = true;

                if (_persistence != null && _persistence.HasActivePackage)
                    await _persistence.SaveArtifactAsync(ArtifactType.EnhancedMesh, null, enhanced);

                MeshEnhanceStatus = "Done";
                Logger.Info($"Mesh enhancement complete: {enhanced.Positions.Length} verts");
            }
            catch (Exception e)
            {
                Logger.Error($"Mesh enhancement failed: {e.Message}\n{e.StackTrace}");
                MeshEnhanceStatus = "Failed";
            }
            finally
            {
                IsMeshEnhancing = false;
            }
        }

        private static byte[] SerializeRefinedMesh(RefinedTextureResult r)
        {
            int vertCount = r.Positions.Length;
            int idxCount = r.Indices.Length;
            int size = 24 + vertCount * 32 + idxCount * 4;

            using var ms = new MemoryStream(size);
            using var w = new BinaryWriter(ms);

            w.Write((uint)0x46524D52); // RMRF magic
            w.Write(1);                // version
            w.Write(vertCount);
            w.Write(idxCount);
            w.Write(r.AtlasWidth);
            w.Write(r.AtlasHeight);

            for (int i = 0; i < vertCount; i++)
            {
                w.Write(r.Positions[i].x); w.Write(r.Positions[i].y); w.Write(r.Positions[i].z);
                w.Write(r.Normals[i].x); w.Write(r.Normals[i].y); w.Write(r.Normals[i].z);
                w.Write(r.UVs[i].x); w.Write(r.UVs[i].y);
            }
            foreach (int idx in r.Indices) w.Write(idx);

            return ms.ToArray();
        }

        private static RefinedTextureResult DeserializeRefinedMesh(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var r = new BinaryReader(ms);

            uint magic = r.ReadUInt32();
            if (magic == 0x46524D52) r.ReadInt32(); // skip version
            else ms.Position = 4; // no header, rewind past magic (treat as vertCount)

            int vertCount = magic == 0x46524D52 ? r.ReadInt32() : (int)magic;
            int idxCount = r.ReadInt32();
            int atlasW = r.ReadInt32();
            int atlasH = r.ReadInt32();

            var positions = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];
            for (int i = 0; i < vertCount; i++)
            {
                positions[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                normals[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                uvs[i] = new Vector2(r.ReadSingle(), r.ReadSingle());
            }

            int[] indices = new int[idxCount];
            for (int i = 0; i < idxCount; i++) indices[i] = r.ReadInt32();

            return new RefinedTextureResult
            {
                Positions = positions, Normals = normals, UVs = uvs,
                Indices = indices, AtlasWidth = atlasW, AtlasHeight = atlasH,
            };
        }

        private void ApplyEnhancedMesh(RefinedTextureResult enhanced)
        {
            if (_refinedMesh == null)
            {
                _refinedMesh = new Mesh { name = "EnhancedScanMesh", indexFormat = IndexFormat.UInt32 };
            }

            _refinedMesh.Clear();
            _refinedMesh.SetVertices(enhanced.Positions);
            _refinedMesh.SetNormals(enhanced.Normals);
            _refinedMesh.SetUVs(0, enhanced.UVs);
            _refinedMesh.SetTriangles(enhanced.Indices, 0);
            _refinedMesh.RecalculateBounds();

            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = _refinedMesh;

            if (_refinedAtlasTexture != null)
                _refinedRenderer.material.mainTexture = _refinedAtlasTexture;
        }

        /// <summary>
        /// Build the refined textures and mesh over a few frames: two 19 MB
        /// texture uploads, a mesh upload and a tangent pass in one frame was
        /// the hitch at the end of every refinement. Tangents come from a worker.
        /// </summary>
        private async Task ApplyRefinedAtlasAsync(RefinedTextureResult result)
        {
            var tangentTask = Task.Run(() => TextureRefinement.ComputeTangents(
                result.Positions, result.Normals, result.UVs, result.Indices));

            if (_refinedAtlasTexture != null)
                Destroy(_refinedAtlasTexture);

            _refinedAtlasTexture = new Texture2D(result.AtlasWidth, result.AtlasHeight,
                TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear };
            _refinedAtlasTexture.SetPixelData(result.AtlasPixels, 0);
            _refinedAtlasTexture.Apply();
            await Task.Yield();

            if (_normalMapTexture != null)
                Destroy(_normalMapTexture);
            _normalMapTexture = null;
            if (result.NormalPixels != null)
            {
                _normalMapTexture = new Texture2D(result.AtlasWidth, result.AtlasHeight,
                    TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear };
                _normalMapTexture.SetPixelData(result.NormalPixels, 0);
                _normalMapTexture.Apply();
                await Task.Yield();
            }

            if (_refinedMesh == null)
                _refinedMesh = new Mesh { name = "RefinedScanMesh", indexFormat = IndexFormat.UInt32 };

            var tangents = await tangentTask;
            _refinedMesh.Clear();
            _refinedMesh.SetVertices(result.Positions);
            _refinedMesh.SetNormals(result.Normals);
            _refinedMesh.SetUVs(0, result.UVs);
            _refinedMesh.SetTangents(tangents);
            _refinedMesh.SetTriangles(result.Indices, 0);
            _refinedMesh.RecalculateBounds();

            Logger.Info($"Refined mesh applied: " +
                $"{result.Positions.Length} verts, {result.Indices.Length / 3} tris, " +
                $"atlas {result.AtlasWidth}x{result.AtlasHeight}" +
                (result.NormalPixels != null ? " +normal" : ""));

            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = _refinedMesh;
            _refinedRenderer.material.mainTexture = _refinedAtlasTexture;
            if (_normalMapTexture != null)
                _refinedRenderer.material.SetTexture("_BumpMap", _normalMapTexture);
        }

        /// <summary>
        /// Applies pre-loaded atlas and mesh data (called from persistence load).
        /// </summary>
        internal void ApplyRefinedTexture(Texture2D atlas, Mesh mesh, Texture2D normalMap = null)
        {
            _refinedAtlasTexture = atlas;
            _refinedMesh = mesh;
            if (_normalMapTexture != null)
                Destroy(_normalMapTexture);
            _normalMapTexture = normalMap;

            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = mesh;
            _refinedRenderer.material.mainTexture = atlas;
            if (_normalMapTexture != null)
                _refinedRenderer.material.SetTexture("_BumpMap", _normalMapTexture);
            HasRefinedTexture = true;
            PopulateSceneObjectRegistry();
            RefinedMeshReady?.Invoke(mesh, atlas);
        }

        /// <summary>Sets the SceneObjectRegistry (used by persistence load).</summary>
        internal void SetSceneObjectRegistry(SceneObjectRegistry registry)
        {
            _sceneObjectRegistry = registry;
        }

        /// <summary>
        /// Populates the SceneObjectRegistry from live MRUK anchors.
        /// Always replaces stale MRUK entries with fresh tracking-accurate positions.
        /// Safe to call multiple times — AI detections are preserved.
        /// </summary>
        internal void PopulateSceneObjectRegistry()
        {
            _sceneObjectRegistry ??= new SceneObjectRegistry();
            if (_roomUnderstanding == null) return;

            _roomUnderstanding.RefreshRoom();
            _sceneObjectRegistry.RemoveBySource(SceneObjectSource.MRUK);
            _roomUnderstanding.PopulateRegistry(_sceneObjectRegistry);
        }

        private void SubscribeToAnchorsChanged()
        {
            if (_subscribedToAnchorsChanged || _roomUnderstanding == null) return;
            _roomUnderstanding.AnchorsChanged += OnMrukAnchorsChanged;
            _subscribedToAnchorsChanged = true;
        }

        private void UnsubscribeFromAnchorsChanged()
        {
            if (!_subscribedToAnchorsChanged || _roomUnderstanding == null) return;
            _roomUnderstanding.AnchorsChanged -= OnMrukAnchorsChanged;
            _subscribedToAnchorsChanged = false;
        }

        private void OnMrukAnchorsChanged()
        {
            Logger.Info("[RoomScanner] MRUK anchors changed — re-populating registry");
            PopulateSceneObjectRegistry();
            if (IsScanning) BindScanPriors();
        }

        void ResolveScanRoomUuid()
        {
            if (_scanRoomUuid != Guid.Empty) return;
            if (_roomUnderstanding == null) return;
            _scanRoomUuid = _roomUnderstanding.TryGetRoomUuidContainingHeadset();
        }

        void BindScanPriors()
        {
            if (_volumeIntegrator == null)
                return;
            if (_roomUnderstanding == null)
            {
                _volumeIntegrator.ClearScanPriors();
                return;
            }

            ResolveScanRoomUuid();
            _roomUnderstanding.CopyRoomClipPlanes(_scanRoomUuid, _clipScratch);
            if (stampScreenPlanes)
                _roomUnderstanding.CopyScreenStamps(_scanRoomUuid, _stampScratch);
            else
                _stampScratch.Clear();
            bool useAabb = false;
            Vector3 aabbMin = Vector3.zero, aabbMax = Vector3.zero;
            if (confineScanToContainingRoom)
                useAabb = _roomUnderstanding.CopyRoomWorldAabb(
                    _scanRoomUuid, out aabbMin, out aabbMax);
            _volumeIntegrator.SetScanPriors(
                confineScanToContainingRoom, _clipScratch, _stampScratch,
                useAabb, aabbMin, aabbMax);

            if (_stampScratch.Count > 0 || (confineScanToContainingRoom && _clipScratch.Count > 0))
            {
                Logger.Info(
                    $"[RoomScanner] Scan priors — confine={confineScanToContainingRoom} " +
                    $"room={_scanRoomUuid} clipPlanes={_clipScratch.Count} " +
                    $"aabb={(useAabb ? 1 : 0)} screens={_stampScratch.Count}");
            }

            BindShellCells();
        }

        void BindShellCells()
        {
            if (_volumeIntegrator == null || _shellTracker == null) return;
            int uploaded = _roomUnderstanding != null
                ? _roomUnderstanding.CopyShellCells(_scanRoomUuid, _shellCells)
                : 0;
            _volumeIntegrator.SetShellCells(_shellCells.GpuPos, _shellCells.GpuNrm, uploaded);
            _shellTracker.Reset();
            if (uploaded > 0)
            {
                Logger.Info(
                    $"[RoomScanner] Shell cells: total={_shellCells.CellCount} uploaded={uploaded} " +
                    $"excluded={_shellCells.ExcludedCount} cell={_shellCells.CellSize:F2}m " +
                    $"surfaces={_shellCells.SurfaceCount} dropped={_shellCells.DroppedSurfaces}");
            }
        }

        void OnShellResult(uint[] result, int count, int generation)
        {
            if (!IsScanning || _shellTracker == null || _volumeIntegrator == null) return;
            if (generation != _volumeIntegrator.ShellGeneration) return;
            _shellTracker.Update(result, count, _volumeIntegrator, _volumeIntegrator.VoxelSize);

            float t = Time.time;
            if (t - _lastShellLog >= 2f)
            {
                _lastShellLog = t;
                var g = _shellTracker.LargestGap;
                Logger.Verbose(
                    $"[RoomScanner] Shell coverage {_shellTracker.Coverage:P0} " +
                    $"({_shellTracker.Covered}/{_shellTracker.Uploaded - _shellTracker.Empty}, " +
                    $"empty={_shellTracker.Empty}) gaps={_shellTracker.GapCount} " +
                    $"largest={g.Cells} cells ({g.Kind}) fills={_shellTracker.FillsApplied}");
            }
        }

        /// <summary>
        /// Largest uncovered shell patches, largest first (at most 8). Zero
        /// when shell coverage is unavailable.
        /// </summary>
        public int CopyShellGaps(List<ShellGap> dest)
        {
            if (_shellTracker == null)
            {
                dest?.Clear();
                return 0;
            }
            return _shellTracker.CopyGaps(dest);
        }

        void MaybeRetryScanRoomBind()
        {
            if (!confineScanToContainingRoom) return;
            if (_roomUnderstanding == null) return;
            if (_scanRoomUuid != Guid.Empty) return;

            float now = Time.time;
            if (now - _lastEmptyRoomBindAttempt < 2f) return;
            _lastEmptyRoomBindAttempt = now;
            BindScanPriors();
        }

        internal void ApplyHQTexture(Texture2D atlas)
        {
            _hqAtlasTexture = atlas;
            EnsureRefinedRenderer();
            HasHQRefinedTexture = true;
        }

        /// <summary>
        /// Returns the cached UV-unwrapped mesh, or runs the unwrap if not yet done.
        /// Shared by both on-device refine and HQ refine paths.
        /// </summary>
        private async Task<UnwrappedMeshResult> EnsureUnwrappedAsync()
        {
            if (_cachedUnwrap.HasValue)
            {
                Logger.Info("Reusing cached UV-unwrapped mesh");
                return _cachedUnwrap.Value;
            }

            // Reconstruct from persisted refined mesh if available (skip xatlas)
            if (LastRefinedResult.HasValue)
            {
                var r = LastRefinedResult.Value;
                var unwrap = ReconstructUnwrapFromResult(r);
                _cachedUnwrap = unwrap;
                EnsureRefinedMesh(unwrap);
                Logger.Info("Reconstructed UV mesh from persisted refined_mesh.bin (no xatlas needed)");
                return unwrap;
            }

            string kfDir = KeyframeDirectory;
            var unwrap2 = await _textureRefinement.UnwrapMeshAsync(kfDir, KeyframeRelocation);
            _cachedUnwrap = unwrap2;

            EnsureRefinedMesh(unwrap2);

            LastRefinedResult = new RefinedTextureResult
            {
                Positions = unwrap2.Positions,
                Normals = unwrap2.Normals,
                UVs = unwrap2.UVs,
                Indices = unwrap2.Indices,
                AtlasWidth = unwrap2.AtlasWidth,
                AtlasHeight = unwrap2.AtlasHeight,
                AtlasPixels = null,
            };

            return unwrap2;
        }

        /// <summary>
        /// Rebuilds an UnwrappedMeshResult from a persisted RefinedTextureResult.
        /// RawUVs are reconstructed from normalized UVs * atlas dimensions.
        /// OrigPositions/OrigIndices use the unwrapped mesh as fallback (sufficient for depth buffer).
        /// </summary>
        private static UnwrappedMeshResult ReconstructUnwrapFromResult(RefinedTextureResult r)
        {
            int vertCount = r.Positions.Length;
            float[] rawUVs = new float[vertCount * 2];
            for (int i = 0; i < vertCount; i++)
            {
                rawUVs[i * 2] = r.UVs[i].x * r.AtlasWidth;
                rawUVs[i * 2 + 1] = r.UVs[i].y * r.AtlasHeight;
            }

            return new UnwrappedMeshResult
            {
                Positions = r.Positions,
                Normals = r.Normals,
                UVs = r.UVs,
                RawUVs = rawUVs,
                Indices = r.Indices,
                AtlasWidth = r.AtlasWidth,
                AtlasHeight = r.AtlasHeight,
                OrigPositions = r.Positions,
                OrigNormals = r.Normals,
                OrigIndices = r.Indices,
            };
        }

        private void EnsureRefinedMesh(UnwrappedMeshResult unwrap)
        {
            if (_refinedMesh != null) return;
            _refinedMesh = new Mesh { name = "RefinedScanMesh", indexFormat = IndexFormat.UInt32 };
            _refinedMesh.SetVertices(unwrap.Positions);
            _refinedMesh.SetNormals(unwrap.Normals);
            _refinedMesh.SetUVs(0, unwrap.UVs);
            _refinedMesh.SetTriangles(unwrap.Indices, 0);
            // Without explicit bounds, Unity's frustum culling treats the
            // mesh as a zero-size point at the local origin and the renderer
            // pops in/out depending on view direction. SetTriangles does not
            // recompute bounds when only positions/indices change.
            _refinedMesh.RecalculateBounds();
            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = _refinedMesh;
        }

        private void EnsureRefinedRenderer()
        {
            if (_refinedRenderer != null) return;

            var go = new GameObject("RefinedMeshRenderer");
            go.transform.SetParent(transform, false);
            _refinedMeshFilter = go.AddComponent<MeshFilter>();
            _refinedRenderer = go.AddComponent<MeshRenderer>();

            var shader = _textureRefinement != null ? _textureRefinement.refinedMeshShader : null;
            if (shader == null)
            {
                Logger.Error("RefinedMesh shader not assigned. Run Setup Wizard to fix.");
                return;
            }
            _refinedMaterial = new Material(shader);
            _refinedRenderer.material = _refinedMaterial;
            _refinedRenderer.enabled = false;

            var occShader = _textureRefinement != null ? _textureRefinement.occlusionMeshShader : null;
            if (occShader != null)
                _occlusionMaterial = new Material(occShader);
        }

        /// <summary>
        /// Swap the refined mesh between two-sided (in-room) and Cull Back
        /// (outside). Quest ignores ShaderLab <c>Cull [_Cull]</c>; this is a
        /// second program, not a float.
        /// </summary>
        internal void SetRefinedBackfaceCull(bool cullBack)
        {
            if (_refinedMaterial == null || _textureRefinement == null) return;
            var shader = cullBack
                ? _textureRefinement.refinedMeshBackfaceShader
                : _textureRefinement.refinedMeshShader;
            if (shader == null || _refinedMaterial.shader == shader) return;
            _refinedMaterial.shader = shader;
        }

        // ─────────────────────────────────────────────────────────────
        //  Internal helpers
        // ─────────────────────────────────────────────────────────────

        private void SetupHeadExclusion()
        {
            if (_volumeIntegrator == null) return;
            if (_volumeIntegrator.HeadAnchor != null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                _volumeIntegrator.HeadAnchor = cam.transform;
                Logger.Info($"Head exclusion anchor: {cam.gameObject.name}");
            }
            else
            {
                Logger.Warning("No main camera found for head exclusion");
            }
        }

        void RefreshBodyAnchors()
        {
            if (_volumeIntegrator == null || _volumeIntegrator.BodyAnchorsHostOwned)
                return;

            if (_bodyRig == null)
                _bodyRig = FindAnyObjectByType<OVRCameraRig>();

            var rig = _bodyRig;
            if (rig != null && rig.centerEyeAnchor != null)
                _volumeIntegrator.HeadAnchor = rig.centerEyeAnchor;
            else if (_volumeIntegrator.HeadAnchor == null && Camera.main != null)
                _volumeIntegrator.HeadAnchor = Camera.main.transform;

            if (rig == null) return;
            _volumeIntegrator.LeftHandAnchor = PickWrist(rig, left: true);
            _volumeIntegrator.RightHandAnchor = PickWrist(rig, left: false);
        }

        Transform PickWrist(OVRCameraRig rig, bool left)
        {
            var controller = left ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch;
            if (OVRInput.GetControllerPositionTracked(controller))
            {
                var onCtrl = left ? rig.leftHandOnControllerAnchor : rig.rightHandOnControllerAnchor;
                if (onCtrl != null) return onCtrl;
                return left ? rig.leftControllerAnchor : rig.rightControllerAnchor;
            }

            var handAnchor = left ? rig.leftHandAnchor : rig.rightHandAnchor;
            if (handAnchor == null) return null;
            var hand = left ? _leftOvrHand : _rightOvrHand;
            if (hand == null)
            {
                hand = handAnchor.GetComponentInChildren<OVRHand>();
                if (left) _leftOvrHand = hand;
                else _rightOvrHand = hand;
            }
            if (hand != null && !hand.IsTracked) return null;
            return handAnchor;
        }

        // ─────────────────────────────────────────────────────────────
        //  Coverage metrics & progress
        // ─────────────────────────────────────────────────────────────

        private ScanCoverage BuildCoverage()
        {
            int surfaceVoxels = _volumeIntegrator != null ? _volumeIntegrator.SurfaceVoxelCount : 0;
            int frozenVoxels = _volumeIntegrator != null ? _volumeIntegrator.FrozenSurfaceCount : 0;
            int coloredVoxels = _volumeIntegrator != null ? _volumeIntegrator.ColoredSurfaceCount : 0;
            int vertCount = _meshExtractor != null ? _meshExtractor.LastVertexCount : 0;
            int idxCount = _meshExtractor != null ? _meshExtractor.LastIndexCount : 0;

            float colorCov = surfaceVoxels > 0 ? (float)coloredVoxels / surfaceVoxels : 0f;
            float frozenFrac = surfaceVoxels > 0 ? (float)frozenVoxels / surfaceVoxels : 0f;

            var cov = new ScanCoverage
            {
                IntegrationCount = _volumeIntegrator != null ? _volumeIntegrator.IntegrationCount : 0,
                MeshVertexCount = vertCount,
                MeshTriangleCount = idxCount / 3,
                SurfaceVoxelCount = surfaceVoxels,
                FrozenSurfaceCount = frozenVoxels,
                ColoredSurfaceCount = coloredVoxels,
                ColorCoverage = colorCov,
                FrozenFraction = frozenFrac,
                KeyframeCount = _keyframeCollector != null ? _keyframeCollector.SavedCount : 0
            };

            if (_volumeIntegrator != null && _volumeIntegrator.AnalysisAvailable)
            {
                var cl = _volumeIntegrator.Closure;
                cov.AnalysisAvailable = true;
                cov.Closure = cl.Closure;
                cov.Refinement = _volumeIntegrator.Refinement;
                cov.ConfidentFraction = _volumeIntegrator.ConfidentFraction;
                cov.ConfidentSurfaceCount = _volumeIntegrator.ConfidentSurfaceCount;
                cov.LeakAreaM2 = cl.LeakAreaM2;
                cov.SurfaceAreaM2 = cl.SurfaceAreaM2;
                cov.HoleCount = cl.HoleCount;
                cov.LargestHole = cl.LargestHole;
                cov.LeakFills = _volumeIntegrator.LeakFills;
            }

            if (_shellTracker != null && _shellTracker.Available)
            {
                cov.ShellCoverageAvailable = true;
                cov.ShellCoverage = _shellTracker.Coverage;
                cov.ShellCellsTotal = _shellTracker.Uploaded;
                cov.ShellCellsCovered = _shellTracker.Covered;
                cov.ShellCellsExcluded = _shellTracker.Excluded;
                cov.ShellCellsEmpty = _shellTracker.Empty;
                cov.ShellGapCount = _shellTracker.GapCount;
                cov.LargestGap = _shellTracker.LargestGap;
                cov.ShellFillsApplied = _shellTracker.FillsApplied;
            }
            return cov;
        }

        /// <summary>
        /// Progress is analytic only: the mesh's own closure and how much of
        /// the surface has stopped moving. The shell prior is reported beside
        /// it for guidance and drives auto-fill, but never this number.
        /// </summary>
        private ScanProgress BuildProgress()
        {
            var cov = BuildCoverage();
            float progress = cov.AnalysisAvailable ? _volumeIntegrator.Progress : 0f;

            ScanPhase phase;
            if (!IsScanning && _volumeIntegrator != null && _volumeIntegrator.IntegrationCount == 0)
                phase = ScanPhase.NotStarted;
            else
                phase = progress < 0.30f ? ScanPhase.Discovering
                    : progress < 0.90f ? ScanPhase.Refining
                    : progress < 0.95f ? ScanPhase.Stabilized
                    : ScanPhase.Complete;

            return new ScanProgress
            {
                Coverage = cov,
                OverallProgress = Mathf.Clamp01(progress),
                Phase = phase
            };
        }

        private static readonly int NormalFallbackID = Shader.PropertyToID("_RSNormalFallback");

        private void SetSafeShaderDefaults()
        {
            Shader.SetGlobalFloat(TriAvailableID, 0f);
            Shader.SetGlobalFloat(NormalFallbackID, 0f);
            Shader.SetGlobalFloat(WireframeID, 0f);
            Shader.SetGlobalFloat(WireThicknessID, wireThickness);
            Shader.SetGlobalFloat(NoFreezeTintID, showFreezeTint ? 0f : 1f);
            Shader.SetGlobalFloat(ShowHolesID, _showHoles ? 1f : 0f);
        }

        private int _colorFrameLog;
        private bool _cameraAvailable;
        private void ProvideColorFrame()
        {
            ICameraProvider provider = GetActiveCameraProvider();

            // IsPlaying = camera subsystem running (stable signal).
            // IsReady = new frame available this tick (toggles at camera fps < app fps).
            bool cameraPlaying = provider != null && provider.IsPlaying;

            if (cameraPlaying && !_cameraAvailable)
            {
                _cameraAvailable = true;
                Shader.SetGlobalFloat(NormalFallbackID, 0f);
                Logger.Info("Camera playing — disabling normal fallback");
            }
            else if (!cameraPlaying && (_cameraAvailable || _colorFrameLog == 0))
            {
                _cameraAvailable = false;
                Shader.SetGlobalFloat(NormalFallbackID, 1f);
                Logger.Info("Camera not playing — enabling normal fallback rendering");
            }

            if (provider != null && provider.IsReady)
            {
                Texture frame = provider.CurrentFrame;
                if (frame != null)
                {
                    _depthCapture?.SetRGBGuide(frame);

                    Pose pose = provider.CameraPose;
                    if (_depthCapture != null)
                        pose = _depthCapture.TrackingToWorld(pose);
                    Vector2 focal = provider.FocalLength;
                    Vector2 principal = provider.PrincipalPoint;
                    Vector2 sensor = provider.SensorResolution;
                    Vector2 current = provider.CurrentResolution;

                    _volumeIntegrator.SetCameraData(
                        frame, pose.position, pose.rotation,
                        focal, principal, sensor, current);

                    ColorFrameProvided?.Invoke(frame, new Pose(pose.position, pose.rotation),
                        focal, principal, sensor, current);

                    _colorFrameLog++;
                    if (_colorFrameLog <= 3 || _colorFrameLog % 50 == 0)
                        Logger.Verbose($"ColorFrame #{_colorFrameLog}: " +
                            $"frame={frame.width}x{frame.height}");

                    return;
                }
            }

            _colorFrameLog++;
            if (_colorFrameLog <= 5)
                Logger.Verbose($"ColorFrame #{_colorFrameLog}: NO FRAME " +
                    $"playing={cameraPlaying}, " +
                    $"isReady={provider?.IsReady ?? false}");

            _volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private static readonly int NoFreezeTintID = Shader.PropertyToID("_RSNoFreezeTint");
        private static readonly int ShowHolesID = Shader.PropertyToID("_RSShowHoles");
        private static readonly int TriAvailableID = Shader.PropertyToID("_RSTriAvailable");
        private static readonly int WireframeID = Shader.PropertyToID("_RSWireframe");
        private static readonly int WireThicknessID = Shader.PropertyToID("_RSWireThickness");

        private void ApplyRenderMode()
        {
            var gpuRenderer = _meshExtractor != null ? _meshExtractor.GetComponent<GPUMeshRenderer>() : null;

            bool gpuMeshVisible = renderMode == ScanRenderMode.Vertex
                               || renderMode == ScanRenderMode.Triplanar
                               || renderMode == ScanRenderMode.Wireframe;
            bool refinedVisible = renderMode == ScanRenderMode.Refined
                               || renderMode == ScanRenderMode.HQRefined
                               || renderMode == ScanRenderMode.Occlusion;

            if (gpuRenderer != null)
                gpuRenderer.RenderVisible = gpuMeshVisible;
            if (_gsplatProvider != null)
                _gsplatProvider.RenderVisible = renderMode == ScanRenderMode.Splat;
            if (_refinedRenderer != null)
            {
                _refinedRenderer.enabled = refinedVisible;
                if (refinedVisible)
                {
                    if (renderMode == ScanRenderMode.Occlusion && _occlusionMaterial != null)
                    {
                        _refinedRenderer.material = _occlusionMaterial;
                    }
                    else
                    {
                        _refinedRenderer.material = _refinedMaterial;
                        var tex = renderMode == ScanRenderMode.HQRefined && _hqAtlasTexture != null
                            ? _hqAtlasTexture
                            : _refinedAtlasTexture;
                        if (tex != null)
                            _refinedMaterial.mainTexture = tex;
                        if (_normalMapTexture != null)
                            _refinedMaterial.SetTexture("_BumpMap", _normalMapTexture);
                    }
                }
            }

            // Vertex mode: force triplanar off; Triplanar mode: reassert globals from cache
            if (renderMode == ScanRenderMode.Vertex)
                Shader.SetGlobalFloat(TriAvailableID, 0f);
            else if (renderMode == ScanRenderMode.Triplanar && _triplanarCache != null)
                _triplanarCache.UpdateShaderGlobals();

            Shader.SetGlobalFloat(WireframeID, renderMode == ScanRenderMode.Wireframe ? 1f : 0f);
            Shader.SetGlobalFloat(WireThicknessID, wireThickness);
            Shader.SetGlobalFloat(NoFreezeTintID, showFreezeTint ? 0f : 1f);
        }

        /// <summary>
        /// Whether a trained splat has been downloaded and is ready to load/render.
        /// </summary>
        public bool HasDownloadedSplat => _downloadedPlyData != null && _downloadedPlyData.Length > 0;

        private byte[] _downloadedPlyData;

        /// <summary>
        /// Raw PLY bytes from server training or loaded from disk.
        /// Used by persistence to save/restore Gaussian Splat data.
        /// </summary>
        public byte[] DownloadedPlyData
        {
            get => _downloadedPlyData;
            set => _downloadedPlyData = value;
        }

        /// <summary>
        /// Keyframe directory within the active package (tmp or saved).
        /// Returns null if no package is active.
        /// </summary>
        public string KeyframeDirectory
        {
            get
            {
                if (_persistence == null || !_persistence.HasActivePackage) return null;
                return Path.Combine(_persistence.ActivePackageDirectory, "keyframes");
            }
        }

        /// <summary>
        /// Deletes the artifact matching the current render mode from the active package.
        /// </summary>
        public void DeleteActiveArtifact()
        {
            if (_persistence == null || !_persistence.HasActivePackage) return;

            switch (renderMode)
            {
                case ScanRenderMode.Splat:
                    _persistence.DeleteArtifactFromPackage(ArtifactType.Splat);
                    _gsplatProvider?.ClearSplat();
                    _downloadedPlyData = null;
                    SetRenderMode(ScanRenderMode.Vertex);
                    break;

                case ScanRenderMode.Refined:
                    if (HasEnhancedMesh)
                    {
                        DeleteEnhancedMesh();
                    }
                    else
                    {
                        _persistence.DeleteArtifactFromPackage(ArtifactType.Refined);
                        HasRefinedTexture = false;
                        LastRefinedResult = null;
                        LastSimplifiedResult = null;
                        _cachedUnwrap = null;
                        _refinedMesh = null;
                        SetRenderMode(ScanRenderMode.Vertex);
                    }
                    break;

                case ScanRenderMode.HQRefined:
                    _persistence.DeleteArtifactFromPackage(ArtifactType.HQRefined);
                    HasHQRefinedTexture = false;
                    _hqAtlasTexture = null;
                    SetRenderMode(HasRefinedTexture ? ScanRenderMode.Refined : ScanRenderMode.Vertex);
                    break;
            }
        }

        /// <summary>
        /// Removes the enhanced mesh overlay and restores the original refined mesh geometry.
        /// </summary>
        public void DeleteEnhancedMesh()
        {
            if (!HasEnhancedMesh) return;

            _persistence?.DeleteArtifactFromPackage(ArtifactType.EnhancedMesh);
            HasEnhancedMesh = false;

            var restore = LastSimplifiedResult ?? LastRefinedResult;
            if (restore.HasValue)
            {
                var r = restore.Value;
                if (_refinedMesh != null)
                {
                    _refinedMesh.Clear();
                    _refinedMesh.SetVertices(r.Positions);
                    _refinedMesh.SetNormals(r.Normals);
                    _refinedMesh.SetUVs(0, r.UVs);
                    _refinedMesh.SetTriangles(r.Indices, 0);
                    _refinedMesh.RecalculateBounds();
                    EnsureRefinedRenderer();
                    _refinedMeshFilter.mesh = _refinedMesh;
                }
            }

            Logger.Info("Enhanced mesh deleted, restored to " +
                        (LastSimplifiedResult.HasValue ? "simplified" : "original refined"));
        }

        /// <summary>
        /// Loads the downloaded splat into GPU buffers and switches to splat render mode.
        /// Call after training + download completes.
        /// </summary>
        public void LoadDownloadedSplat()
        {
            if (_downloadedPlyData == null || _downloadedPlyData.Length == 0)
            {
                Logger.Warning("No downloaded splat data to load");
                return;
            }
            _gsplatProvider?.LoadTrainedPly(_downloadedPlyData);
            SetRenderMode(ScanRenderMode.Splat);
            Logger.Info("Trained Gaussians loaded and rendering");
        }

        private async void RunServerTrainingAsync()
        {
            if (_serverTrainingInProgress) return;
            _serverTrainingInProgress = true;

            try
            {
                if (_gsplatProvider == null)
                {
                    Logger.Error("No GSplat provider available");
                    return;
                }

                Logger.Info("Starting server-side GS training pipeline...");
                byte[] plyData = await _gsplatProvider.RunServerTrainingAsync(KeyframeDirectory, KeyframeRelocation);
                if (plyData == null || plyData.Length == 0) return;

                _downloadedPlyData = plyData;

                if (_persistence != null && _persistence.HasActivePackage)
                    await _persistence.SaveArtifactAsync(ArtifactType.Splat, plyData);

                LoadDownloadedSplat();
            }
            catch (Exception e)
            {
                Logger.Error($"Server training pipeline error: {e.Message}\n{e.StackTrace}");
            }
            finally
            {
                _serverTrainingInProgress = false;
            }
        }

        private ICameraProvider GetActiveCameraProvider()
        {
            if (_customCameraProvider != null) return _customCameraProvider;
            return _cameraProvider;
        }

    }
}
