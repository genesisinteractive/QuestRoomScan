using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Result returned by <see cref="RoomScanSession"/> operations.
    /// Contains the refined mesh, atlas texture, and the package ID used for persistence.
    /// </summary>
    public struct ScanResult
    {
        /// <summary>The game-ready mesh in world space, relocated with the
        /// anchor pose sampled when the anchor localized. Render this.</summary>
        public Mesh Mesh;
        public Texture2D Atlas;
        public string PackageId;

        /// <summary>
        /// The same mesh in the spatial-anchor frame, rebuilt from package
        /// constants only (see
        /// <see cref="RoomScanPersistence.BuildAnchorFrameMesh"/>). Use this,
        /// not <c>Mesh × root.worldToLocal</c>, for anything you generate once
        /// and persist relative to the room: it is identical across loads and
        /// sessions, where the world mesh moves with tracking. Present the
        /// result under a root bound to the anchor (<see cref="RoomSpaceRoot"/>).
        /// Null when no persistence component is present.
        /// </summary>
        public Mesh AnchorFrameMesh;
    }

    /// <summary>
    /// High-level facade for game developers who need a simple scan → mesh → render workflow.
    /// Wraps <see cref="RoomScanner"/> and <see cref="RoomScanPersistence"/> into a
    /// minimal API with awaitable operations and a single result type.
    /// <para>
    /// Attach this component alongside <see cref="RoomScanner"/> or access via
    /// <see cref="Instance"/> after it initializes. The Setup Scene wizard's
    /// "Apply Game-Ready Setup" preset adds it automatically.
    /// </para>
    /// </summary>
    [RequireComponent(typeof(RoomScanner))]
    public class RoomScanSession : MonoBehaviour
    {
        public static RoomScanSession Instance { get; private set; }

        /// <summary>Raised each frame during scanning with the latest progress.</summary>
        public event Action<ScanProgress> ProgressUpdated;

        /// <summary>
        /// Raised when MRUK discovery finishes (<see cref="IsRoomLoaded"/>).
        /// All scene anchors from that load are already on the rooms.
        /// </summary>
        public event Action RoomReady;

        /// <summary>
        /// Raised when a room or scene anchor is created or updated after
        /// the initial load (<c>RoomUpdatedEvent</c>,
        /// <c>AnchorCreatedEvent</c>).
        /// </summary>
        public event Action SceneAnchorsChanged;

        /// <summary>
        /// When true, TSDF stays inside the MRUK room that contained
        /// the headset when the scan started (planes expanded 50 cm
        /// outward, then hard-confined). Default <c>false</c> (unbounded
        /// scan). Set this before <see cref="StartScanAsync"/>. No-op
        /// without <see cref="RoomUnderstanding"/>. SCREEN (TV) plane
        /// stamps still apply either way when that module is present.
        /// </summary>
        public bool ConfineScanToContainingRoom
        {
            get => _scanner != null && _scanner.ConfineScanToContainingRoom;
            set
            {
                if (_scanner != null)
                    _scanner.ConfineScanToContainingRoom = value;
            }
        }

        /// <summary>
        /// When true, MRUK <c>SCREEN</c> (TV) slabs are stamped as analytic
        /// TSDF after Integrate. Default <c>true</c>. Set before
        /// <see cref="StartScanAsync"/>. No-op without
        /// <see cref="RoomUnderstanding"/>.
        /// </summary>
        public bool StampScreenPlanes
        {
            get => _scanner != null && _scanner.StampScreenPlanes;
            set
            {
                if (_scanner != null)
                    _scanner.StampScreenPlanes = value;
            }
        }

        /// <summary>
        /// Tint the live scan mesh red where the surface is open — the same
        /// boundary the closure metric counts, so the player can see what is
        /// holding <c>ScanProgress.OverallProgress</c> down. Off by default.
        /// </summary>
        public bool ShowHoles
        {
            get => _scanner != null && _scanner.ShowHoles;
            set { if (_scanner != null) _scanner.ShowHoles = value; }
        }

        /// <summary>
        /// Soft-stamp the captured plane over small uncovered wall / floor /
        /// ceiling patches while scanning (see <c>ScanCoverage.ShellFillsApplied</c>).
        /// Default on. Furniture close is a separate inspector toggle.
        /// </summary>
        public bool AutoFillShellGaps
        {
            get => _scanner != null && _scanner.VolumeIntegrator != null && _scanner.VolumeIntegrator.AutoFillShellGaps;
            set
            {
                if (_scanner != null && _scanner.VolumeIntegrator != null)
                    _scanner.VolumeIntegrator.AutoFillShellGaps = value;
            }
        }

        /// <summary>
        /// Largest uncovered shell patches (at most 8, largest first) from the
        /// last coverage tick. Zero when <c>ScanCoverage.ShellCoverageAvailable</c>
        /// is false. Whether a scan may finalize is the host's call — read
        /// <c>ScanProgress.Coverage.ShellCoverage</c> from <see cref="ProgressUpdated"/>.
        /// </summary>
        public int CopyShellGaps(List<ShellGap> dest)
        {
            if (_scanner == null)
            {
                dest?.Clear();
                return 0;
            }
            return _scanner.CopyShellGaps(dest);
        }

        /// <summary>
        /// Pin head and wrist transforms used to skip TSDF voxels around the
        /// operator (torso + hand/forearm capsules). Host-owned — the scanner
        /// will not overwrite them from the camera rig. Call before
        /// <see cref="StartScanAsync"/>. Null wrists skip that side.
        /// </summary>
        public void SetBodyExclusionAnchors(Transform head, Transform leftHand, Transform rightHand)
        {
            if (_scanner == null)
            {
                Logger.Error("RoomScanSession: RoomScanner not found");
                return;
            }
            _scanner.SetBodyExclusionAnchors(head, leftHand, rightHand);
        }

        private RoomScanner _scanner;
        private RoomScanPersistence _persistence;
        RoomUnderstanding _understandingHooked;
        RoomAnchorManager _anchorHooked;

        RoomUnderstanding Understanding()
        {
            var u = RoomUnderstanding.Instance != null
                ? RoomUnderstanding.Instance
                : GetComponent<RoomUnderstanding>();
            HookUnderstanding(u);
            return u;
        }

        void HookUnderstanding(RoomUnderstanding u)
        {
            if (u == null || _understandingHooked == u) return;
            if (_understandingHooked != null)
                _understandingHooked.AnchorsChanged -= ForwardSceneAnchorsChanged;
            _understandingHooked = u;
            u.AnchorsChanged += ForwardSceneAnchorsChanged;
        }

        void HookAnchorManager()
        {
            var mgr = RoomAnchorManager.Instance;
            if (mgr == null || _anchorHooked == mgr) return;
            if (_anchorHooked != null)
                _anchorHooked.RoomReady -= ForwardRoomReady;
            _anchorHooked = mgr;
            mgr.RoomReady += ForwardRoomReady;
            if (mgr.IsRoomLoaded)
                ForwardRoomReady();
        }

        void ForwardRoomReady() => RoomReady?.Invoke();
        void ForwardSceneAnchorsChanged() => SceneAnchorsChanged?.Invoke();

        private void Awake()
        {
            Instance = this;
            _scanner = GetComponent<RoomScanner>();
            _persistence = GetComponent<RoomScanPersistence>();
            HookUnderstanding(GetComponent<RoomUnderstanding>());
            HookAnchorManager();
        }

        private void OnEnable()
        {
            HookUnderstanding(Understanding());
            HookAnchorManager();
        }

        private void OnDestroy()
        {
            if (_understandingHooked != null)
            {
                _understandingHooked.AnchorsChanged -= ForwardSceneAnchorsChanged;
                _understandingHooked = null;
            }
            if (_anchorHooked != null)
            {
                _anchorHooked.RoomReady -= ForwardRoomReady;
                _anchorHooked = null;
            }
            if (Instance == this) Instance = null;
        }

        private void Update()
        {
            if (_anchorHooked == null) HookAnchorManager();
            if (_scanner != null && _scanner.IsScanning)
                ProgressUpdated?.Invoke(_scanner.CurrentProgress);
        }

        /// <summary>
        /// Drops the in-memory loaded / refined scan without deleting saved
        /// packages or erasing spatial-anchor UUIDs. Hides the refined mesh,
        /// unbinds the active spatial anchor, and frees scan GPU if it was
        /// allocated. Call this when leaving a loaded scan to start a new
        /// scan in the same session — <see cref="StartScanAsync"/> also
        /// does this on a non-resume start, but hosts usually want the old
        /// mesh gone <i>before</i> the user presses the scan button.
        /// Idempotent. Yields two frames after the spatial-anchor GameObject
        /// is destroyed so the deferred <c>Destroy</c> has run before the
        /// caller creates a new one.
        /// </summary>
        public async Task UnloadActiveScanAsync()
        {
            if (_scanner == null) return;
            _scanner.UnloadActiveScan();
            await Task.Yield();
            await Task.Yield();
        }

        /// <summary>
        /// Begins a new scan session. The room mesh builds in real-time as
        /// the user looks around. Async because <see cref="RoomScanner.StartScanningAsync"/>
        /// stages the ~600 MB GPU bring-up across a few frames before
        /// enabling the passthrough camera and depth sensor (see that
        /// method's docs). Total wall-clock from await to first integrated
        /// frame is ~56 ms,
        /// imperceptible to the user but worth awaiting so callers can
        /// sequence UI feedback ("Scanning…") right after.
        /// <para>
        /// A non-resume start first unloads any previously loaded package
        /// so a second look in the same session does not keep drawing the
        /// old mesh or create a spatial anchor on top of a still-bound one.
        /// </para>
        /// </summary>
        public Task StartScanAsync()
        {
            if (_scanner == null)
            {
                Logger.Error("RoomScanSession: RoomScanner not found");
                return Task.CompletedTask;
            }
            return _scanner.StartScanningAsync();
        }

        /// <summary>
        /// Paints the voxels inside a spotlight cone from the head (half-angle
        /// <see cref="FreezeConeHalfAngle"/>) as "frozen" — they stop receiving
        /// integration updates until the user explicitly
        /// <see cref="UnfreezeInView"/>s them. Frozen voxels count as refined,
        /// so painting a settled region locks its share of progress.
        /// Integration keeps running on un-painted regions. The debug menu
        /// uses this head cone. Hosts with their own emitter should call the
        /// overload.
        /// </summary>
        public void FreezeInView()
        {
            if (_scanner == null) { Logger.Error("RoomScanSession: RoomScanner not found"); return; }
            _scanner.FreezeInView();
        }

        /// <summary>
        /// Freeze voxels inside a host-supplied spotlight cone instead of the
        /// headset gaze. <paramref name="halfAngleDegrees"/> is the cone's
        /// half-angle; <paramref name="maxMetres"/> 0 is unbounded (same as
        /// the head cone).
        /// </summary>
        public void FreezeInView(Vector3 origin, Vector3 direction, float halfAngleDegrees, float maxMetres = 0f)
        {
            if (_scanner == null) { Logger.Error("RoomScanSession: RoomScanner not found"); return; }
            _scanner.FreezeInView(origin, direction, halfAngleDegrees, maxMetres);
        }

        /// <summary>Half-angle, degrees, of the default head freeze cone.</summary>
        public float FreezeConeHalfAngle => _scanner != null ? _scanner.FreezeConeHalfAngle : 15f;

        /// <summary>
        /// See <see cref="RoomScanner.PresentRefinedWhenReady"/>.
        /// </summary>
        public bool PresentRefinedWhenReady
        {
            get => _scanner != null && _scanner.PresentRefinedWhenReady;
            set { if (_scanner != null) _scanner.PresentRefinedWhenReady = value; }
        }

        /// <summary>
        /// Inverse of <see cref="FreezeInView"/>: unfreezes voxels in the same
        /// cone so depth integration can refine them again. Useful when you
        /// painted too aggressively or part of the scan needs re-capturing.
        /// </summary>
        public void UnfreezeInView()
        {
            if (_scanner == null) { Logger.Error("RoomScanSession: RoomScanner not found"); return; }
            _scanner.UnfreezeInView();
        }

        /// <summary>Unfreeze frozen voxels inside a host-supplied spotlight cone.</summary>
        public void UnfreezeInView(Vector3 origin, Vector3 direction, float halfAngleDegrees, float maxMetres = 0f)
        {
            if (_scanner == null) { Logger.Error("RoomScanSession: RoomScanner not found"); return; }
            _scanner.UnfreezeInView(origin, direction, halfAngleDegrees, maxMetres);
        }

        /// <summary>
        /// Stops scanning, runs on-device texture refinement (UV unwrap + atlas bake + simplification),
        /// and saves to a permanent package. When
        /// <see cref="PresentRefinedWhenReady"/> is true (default), also
        /// switches to the refined mesh and releases the live TSDF
        /// (~400-500 MB). When false, the live vertex mesh stays until the
        /// host presents and releases.
        /// Returns a <see cref="ScanResult"/> with the game-ready mesh and atlas.
        /// </summary>
        public async Task<ScanResult> FinalizeScanAsync()
        {
            if (_scanner == null)
                throw new InvalidOperationException("RoomScanSession: RoomScanner not found");

            _scanner.StopScanning();

            if (!_scanner.HasTextureRefinementModule)
                throw new InvalidOperationException(
                    "RoomScanSession: TextureRefinement module required for FinalizeScanAsync");

            if (!_scanner.HasRefinedTexture)
            {
                var tcs = new TaskCompletionSource<bool>();
                void OnReady(Mesh _, Texture2D __) => tcs.TrySetResult(true);
                _scanner.RefinedMeshReady += OnReady;
                _scanner.StartTextureRefinement();

                var timeout = Task.Delay(TimeSpan.FromMinutes(5));
                var completed = await Task.WhenAny(tcs.Task, timeout);
                _scanner.RefinedMeshReady -= OnReady;

                if (completed == timeout)
                    throw new TimeoutException("Texture refinement timed out (5 min)");
            }

            bool saved = await _scanner.SaveScanAsync();
            if (!saved)
                Logger.Warning("RoomScanSession: save failed — result is in memory only");

            if (_scanner.PresentRefinedWhenReady)
                _scanner.ReleaseScanResources();

            return new ScanResult
            {
                Mesh = _scanner.RefinedMesh,
                Atlas = _scanner.RefinedAtlas,
                PackageId = _persistence?.ActivePackageId,
                AnchorFrameMesh = _persistence?.BuildAnchorFrameMesh(_scanner.RefinedMesh)
            };
        }

        /// <summary>
        /// Loads only the refined mesh + atlas from a previously saved package.
        /// Skips TSDF reconstruction — typically completes in under 1 second.
        /// </summary>
        public async Task<ScanResult> LoadAsync(string packageId)
        {
            if (_scanner == null)
                throw new InvalidOperationException("RoomScanSession: RoomScanner not found");

            bool ok = await _scanner.LoadRefinedOnlyAsync(packageId);
            if (!ok)
                throw new InvalidOperationException($"Failed to load package: {packageId}");

            return new ScanResult
            {
                Mesh = _scanner.RefinedMesh,
                Atlas = _scanner.RefinedAtlas,
                PackageId = packageId,
                AnchorFrameMesh = _persistence?.BuildAnchorFrameMesh(_scanner.RefinedMesh)
            };
        }

        /// <summary>
        /// Loads the most recently saved scan package. Convenience wrapper around <see cref="LoadAsync"/>.
        /// </summary>
        public async Task<ScanResult> LoadLatestAsync()
        {
            if (_persistence == null)
                throw new InvalidOperationException("RoomScanSession: RoomScanPersistence not found");

            var packages = _persistence.ListPackages();
            if (packages.Count == 0)
                throw new InvalidOperationException("No saved scan packages found");

            return await LoadAsync(packages[0].id);
        }

        /// <summary>Returns true if at least one saved scan package exists on disk.</summary>
        public bool HasSavedScan => _persistence != null && _persistence.HasAnyPackage();

        /// <summary>
        /// Snapshot of one saved scan package. Games that keep several rooms
        /// (one package per save) use this with <see cref="LoadAsync"/> and
        /// <see cref="DeleteScanAsync"/> instead of <see cref="LoadLatestAsync"/>
        /// / <see cref="ClearAllScansAsync"/>.
        /// </summary>
        public readonly struct SavedScanInfo
        {
            public SavedScanInfo(string id, string displayName, long timestamp,
                string sceneRoomUuid = "")
            {
                Id = id;
                DisplayName = displayName ?? string.Empty;
                Timestamp = timestamp;
                SceneRoomUuid = sceneRoomUuid ?? string.Empty;
            }

            public string Id { get; }
            public string DisplayName { get; }
            /// <summary>Unix seconds (UTC); <see cref="ListSavedScans"/> is newest-first.</summary>
            public long Timestamp { get; }
            /// <summary>Scene API UUID of the MRUK room this package was
            /// scanned in. Empty until the package has been saved or loaded
            /// once after that field existed.</summary>
            public string SceneRoomUuid { get; }
        }

        /// <summary>
        /// Every saved scan package, newest first. Empty when nothing is on
        /// disk. Does not load meshes.
        /// </summary>
        public IReadOnlyList<SavedScanInfo> ListSavedScans()
        {
            if (_persistence == null) return Array.Empty<SavedScanInfo>();
            var packages = _persistence.ListPackages();
            var list = new List<SavedScanInfo>(packages.Count);
            for (int i = 0; i < packages.Count; i++)
            {
                var p = packages[i];
                list.Add(new SavedScanInfo(p.id, p.displayName, p.timestamp, p.sceneRoomUuid));
            }
            return list;
        }

        /// <summary>
        /// Deletes one saved scan package (mesh, atlas, keyframes, triplanar,
        /// manifest entry) and erases its spatial anchor from Horizon OS.
        /// No-op when the id is missing or already gone.
        /// </summary>
        public Task DeleteScanAsync(string packageId)
        {
            if (_persistence == null || string.IsNullOrEmpty(packageId))
                return Task.CompletedTask;
            return _persistence.DeletePackageAsync(packageId);
        }

        /// <summary>
        /// Deletes every saved scan package on disk (mesh, atlas, keyframes,
        /// triplanar, manifest) and erases each package's spatial anchor from
        /// Horizon OS. Nuclear option — games that keep several packages
        /// should call <see cref="DeleteScanAsync"/> for the one they are
        /// replacing. Safe to call when nothing is saved (returns immediately).
        /// </summary>
        public Task ClearAllScansAsync()
        {
            if (_persistence == null) return Task.CompletedTask;
            return _persistence.ClearAllPackagesAsync();
        }

        /// <summary>Whether a scan is currently in progress.</summary>
        public bool IsScanning => _scanner != null && _scanner.IsScanning;

        /// <summary>Releases heavy GPU resources. Called automatically by <see cref="FinalizeScanAsync"/>.</summary>
        public void ReleaseScanResources() => _scanner?.ReleaseScanResources();

        // ─── Camera permission (HEADSET_CAMERA on Quest 3+) ──────────────
        //
        // PCA can technically wait for the user's permission decision in its
        // own coroutine (see Meta.XR.PassthroughCameraAccess.OnEnable), but
        // game code usually wants to surface a deterministic "asking for
        // permission" UI state and only call StartScan() after the user has
        // decided. These helpers expose that without making callers reach
        // into UnityEngine.Android.Permission directly.

        /// <summary>True when the Horizon OS HEADSET_CAMERA permission has
        /// been granted. Always true outside Android device builds.</summary>
        public bool HasCameraPermission => PassthroughCameraProvider.HasCameraPermission;

        /// <summary>Asynchronously requests the HEADSET_CAMERA permission and
        /// resolves once the user accepts, denies, or dismisses the system
        /// dialog. Call this <b>before</b> <see cref="StartScan"/> to avoid
        /// scanning in degraded depth-only mode while the dialog is up.
        /// Resolves <c>true</c> immediately if permission is already granted,
        /// or outside Android device builds.</summary>
        public Task<bool> RequestCameraPermissionAsync()
            => PassthroughCameraProvider.RequestCameraPermissionAsync();

        /// <summary>True when Horizon OS <c>USE_SCENE</c> (spatial data) is
        /// granted. Always true outside Android device builds.</summary>
        public bool HasScenePermission => AndroidRuntimePermission.Has(AndroidRuntimePermission.Scene);

        /// <summary>Requests spatial-data permission. Resolves true if already
        /// granted, or outside Android device builds.</summary>
        public Task<bool> RequestScenePermissionAsync()
            => AndroidRuntimePermission.RequestAsync(AndroidRuntimePermission.Scene);

        /// <summary>True when Horizon OS <c>USE_ANCHOR_API</c> is granted.
        /// Always true outside Android device builds.</summary>
        public bool HasAnchorPermission => AndroidRuntimePermission.Has(AndroidRuntimePermission.Anchors);

        /// <summary>Requests spatial-anchor permission. Resolves true if
        /// already granted, or outside Android device builds.</summary>
        public Task<bool> RequestAnchorPermissionAsync()
            => AndroidRuntimePermission.RequestAsync(AndroidRuntimePermission.Anchors);

        /// <summary>True after MRUK <c>LoadSceneFromDevice</c> finished,
        /// including an empty space. All discovery anchors are present.
        /// Distinct from <see cref="HasSceneRooms"/>.</summary>
        public bool IsRoomLoaded =>
            RoomAnchorManager.Instance != null && RoomAnchorManager.Instance.IsRoomLoaded;

        /// <summary>True when this space has at least one MRUK room after
        /// discovery. False after a finished load with no scene model.
        /// Does not mean the headset is inside that room — see
        /// <see cref="IsHeadsetInsideASceneRoom"/>.</summary>
        public bool HasSceneRooms =>
            RoomAnchorManager.Instance != null && RoomAnchorManager.Instance.HasSceneRooms;

        /// <summary>
        /// True when the headset is inside <b>any</b> loaded captured space
        /// (outer wall planes, including doorway faces). Boot / Space Setup
        /// should use this — any set-up room is enough. A loaded scan is
        /// tied to one room; use <see cref="IsHeadsetInsideBoundSceneRoom"/>
        /// for that. Native <c>IsPositionInRoom</c> alone is the floor
        /// outline and stays true a little past the door. Editor is always true.
        /// </summary>
        public bool IsHeadsetInsideASceneRoom
        {
            get
            {
                if (Application.isEditor) return true;
                var u = Understanding();
                return u != null && u.IsHeadsetInsideAnyRoom();
            }
        }

        /// <summary>
        /// Scene API UUID of the MRUK room the active package was scanned
        /// in (stored next to the spatial-anchor UUID in
        /// <c>anchor.json</c> / the package manifest). Empty when no
        /// package is loaded, or until the first bind. Resolves and
        /// persists from the localized spatial-anchor pose when missing
        /// or stale. Editor does not invent a UUID.
        /// </summary>
        public Guid BoundSceneRoomUuid
        {
            get
            {
                if (_persistence == null || !_persistence.HasActivePackage)
                    return Guid.Empty;
                string s = _persistence.EnsureAndGetSceneRoomUuid();
                return Guid.TryParse(s, out var g) ? g : Guid.Empty;
            }
        }

        /// <summary>
        /// True when the headset is inside the active package's bound
        /// room — not merely some other captured space in the house.
        /// False when no package is loaded or the bound UUID cannot be
        /// resolved. Editor is always true.
        /// </summary>
        public bool IsHeadsetInsideBoundSceneRoom
        {
            get
            {
                if (Application.isEditor)
                    return true;
                if (_persistence == null || !_persistence.HasActivePackage)
                    return false;
                Guid uuid = BoundSceneRoomUuid;
                if (uuid == Guid.Empty) return false;
                var u = Understanding();
                return u != null && u.IsHeadsetInsideRoom(uuid);
            }
        }

        /// <summary>
        /// Vertical Scene API planes of every loaded room that contains
        /// the headset. <paramref name="kind"/> is the labels the host
        /// wants. Empty in the editor and when the headset is not inside
        /// a captured room. Clears <paramref name="dest"/>.
        /// </summary>
        public int CopyHeadsetRoomWallFaces(List<SceneWallFace> dest, SceneFaceKind kind)
        {
            var u = Understanding();
            if (u == null)
            {
                dest?.Clear();
                return 0;
            }
            return u.CopyHeadsetRoomWallFaces(dest, kind);
        }

        /// <summary>
        /// Scene API labels of the captured room that contains the headset:
        /// the names of <c>MRUKAnchor.SceneLabels</c> its anchors carry
        /// (<c>BED</c>, <c>TABLE</c>, <c>COUCH</c>, <c>SCREEN</c>,
        /// <c>WALL_FACE</c>, …), each once. Overlapping captures are unioned.
        /// Empty in the editor and when the headset is not inside a captured
        /// room. Clears <paramref name="dest"/>.
        /// </summary>
        public int CopyHeadsetRoomLabels(List<string> dest)
        {
            var u = Understanding();
            if (u == null)
            {
                dest?.Clear();
                return 0;
            }
            return u.CopyHeadsetRoomLabels(dest);
        }

        readonly List<string> _labelScratch = new List<string>();

        /// <summary>
        /// True when the captured room that contains the headset has an
        /// anchor with this Scene API label (for example <c>"BED"</c>;
        /// case-insensitive). False in the editor, when the headset is not
        /// inside a captured room, and before the Scene API has loaded.
        /// </summary>
        public bool HeadsetRoomHasLabel(string label)
        {
            if (string.IsNullOrEmpty(label)) return false;
            CopyHeadsetRoomLabels(_labelScratch);
            for (int i = 0; i < _labelScratch.Count; i++)
            {
                if (string.Equals(_labelScratch[i], label, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Scene API UUID of the loaded room that contains the headset
        /// (wall-plane test), or empty. Distinct from
        /// <see cref="BoundSceneRoomUuid"/> (the active scan package).
        /// Editor does not invent a UUID.
        /// </summary>
        public Guid HeadsetSceneRoomUuid
        {
            get
            {
                var u = Understanding();
                return u != null ? u.TryGetRoomUuidContainingHeadset() : Guid.Empty;
            }
        }

        /// <summary>
        /// After a package is loaded and its spatial anchor has localized:
        /// true when the headset and that anchor sit in the <b>same</b>
        /// captured room (wall-plane test). Persists that room's current
        /// Scene API UUID — a Space Setup redo in the same physical room
        /// gets a new UUID and must rebind. False when either pose is
        /// outside a captured volume or they disagree (hallway, a different
        /// set-up room). Editor is always true.
        /// </summary>
        public bool TryRebindBoundSceneRoomIfHeadsetMatches()
        {
            if (Application.isEditor)
                return true;
            if (_persistence == null || !_persistence.HasActivePackage)
                return false;
            var u = Understanding();
            if (u == null) return false;

            Guid atHeadset = u.TryGetRoomUuidContainingHeadset();
            Guid atAnchor = Guid.Empty;
            var mgr = RoomAnchorManager.Instance;
            if (mgr != null && mgr.HasSpatialAnchor && mgr.SpatialAnchorTransform != null)
                atAnchor = u.TryGetRoomUuidAt(mgr.SpatialAnchorTransform.position);

            if (atHeadset == Guid.Empty || atAnchor == Guid.Empty || atHeadset != atAnchor)
                return false;

            _persistence.BindSceneRoomUuid(atAnchor);
            return true;
        }

        /// <summary>The refined-mesh renderer for the active scan, or null.</summary>
        public MeshRenderer RefinedMeshRenderer =>
            _scanner != null ? _scanner.RefinedMeshRenderer : null;

        /// <summary>
        /// Two-sided in the room, Cull Back outside so the near walls vanish
        /// and the interior reads as a shell. No-op if shaders are unwired.
        /// </summary>
        public void SetRefinedBackfaceCull(bool cullBack)
        {
            _scanner?.SetRefinedBackfaceCull(cullBack);
        }

        /// <summary>Completes when MRUK <c>LoadSceneFromDevice</c> has
        /// finished. Every scene anchor from that discovery is already on
        /// the rooms. Completed immediately if it already has.</summary>
        public Task WaitUntilRoomReadyAsync()
        {
            var mgr = RoomAnchorManager.Instance;
            if (mgr == null) return Task.CompletedTask;
            return mgr.WaitUntilRoomReadyAsync();
        }

        /// <summary>
        /// Re-run MRUK scene discovery without opening Space Setup. Use after
        /// spatial-data permission is granted: the first boot load often
        /// finished with zero rooms while <c>USE_SCENE</c> was still denied.
        /// </summary>
        public Task<bool> ReloadSceneFromDeviceAsync()
        {
            var mgr = RoomAnchorManager.Instance;
            if (mgr == null) return Task.FromResult(false);
            return mgr.ReloadSceneFromDeviceAsync();
        }

        /// <summary>
        /// Opens Horizon Space Setup (Unity app pauses), then reloads the
        /// scene model with auto-capture off. Returns true only when rooms
        /// exist afterwards — cancel still completes the OS API as true.
        /// Device-only; editor returns the current room flag.
        /// </summary>
        public Task<bool> RequestSpaceSetupAndReloadAsync()
        {
            var mgr = RoomAnchorManager.Instance;
            if (mgr == null) return Task.FromResult(false);
            return mgr.RequestSpaceSetupAndReloadAsync();
        }
    }
}
