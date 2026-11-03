using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using Genesis.RoomScan.UI;
using Meta.XR;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Rendering.Universal;
using UnityEngine.UIElements;
using UnityEngine.XR.ARFoundation;
using Unity.XR.CoreUtils;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard : EditorWindow
    {
        double _lastRefresh;
        const double REFRESH_SEC = 0.8;
        Vector2 _scroll;

        // Cached scene state
        ARSession _arSession;
        AROcclusionManager _arOcclusion;
        GameObject _cameraRig;

        DepthCapture _depthCapture;
        VolumeIntegrator _volumeIntegrator;
        MeshExtractor _meshExtractor;
        RoomScanner _roomScanner;
        PassthroughCameraProvider _cameraProvider;
        PassthroughCameraAccess _pcaComponent;
        CameraDebugOverlay _cameraDebug;
        DepthDebugOverlay _depthDebug;
        TriplanarCache _triplanarCache;
        RoomScanPersistence _persistence;
        RoomScanSession _session;
        KeyframeCollector _keyframeCollector;
        DebugMenuController _debugMenu;
        RoomScanInputHandler _inputHandler;
        RoomAnchorManager _roomAnchor;
        EventSystem _eventSystem;
        OVRInputModule _ovrInputModule;
        VRDocumentRaycaster _vrRaycaster;
        ControllerRayDriver _rayDriver;
        PanelInputConfiguration _panelInputConfig;

        TextureRefinement _textureRefinement;

        bool _depthCaptureWired, _volumeWired, _meshMatWired, _triplanarWired, _computeShaderWired;
        bool _refinedShaderWired, _occlusionShaderWired, _atlasBakeComputeWired;
        bool _debugOverlayWired;
        bool _boundarylessManifest;
        bool _cleartextAllowed;
        bool _insecureHttpAllowed;

        // Style
        static readonly Color COL_OK   = new(0.25f, 0.82f, 0.35f);
        static readonly Color COL_WARN = new(0.95f, 0.78f, 0.15f);
        static readonly Color COL_MISS = new(0.92f, 0.28f, 0.25f);
        static readonly Color COL_INFO = new(0.45f, 0.72f, 0.95f);
        static readonly Color COL_SECT = new(0.18f, 0.18f, 0.22f);

        const string PKG = "Packages/com.genesis.roomscan/Runtime/Shaders/";

        [MenuItem("RoomScan/Setup Scene")]
        static void Open()
        {
            var w = GetWindow<RoomScanSetupWizard>("Room Scan Setup");
            w.minSize = new Vector2(420, 480);
        }

        void OnEnable()  => Refresh();
        void OnFocus()   => Refresh();

        void Update()
        {
            if (EditorApplication.timeSinceStartup - _lastRefresh > REFRESH_SEC)
            {
                Refresh();
                Repaint();
            }
        }

        // =================================================================
        //  REFRESH
        // =================================================================

        void Refresh()
        {
            _lastRefresh = EditorApplication.timeSinceStartup;

            _arSession = FindAny<ARSession>();
            _arOcclusion = FindAny<AROcclusionManager>();

            // Try to find camera rig — look for OVRCameraRig or XROrigin
            _cameraRig = null;
            var xrOrigin = FindAny<Unity.XR.CoreUtils.XROrigin>();
            if (xrOrigin != null)
                _cameraRig = xrOrigin.gameObject;
            if (_cameraRig == null)
            {
                var ovrRig = FindComponentByTypeName("OVRCameraRig");
                if (ovrRig != null) _cameraRig = ovrRig.gameObject;
            }

            _depthCapture = FindAny<DepthCapture>();
            _volumeIntegrator = FindAny<VolumeIntegrator>();
            _meshExtractor = FindAny<MeshExtractor>();
            _roomScanner = FindAny<RoomScanner>();
            _cameraProvider = FindAny<PassthroughCameraProvider>();
            _pcaComponent = FindAny<PassthroughCameraAccess>();
            _cameraDebug = FindAny<CameraDebugOverlay>();
            _depthDebug = FindAny<DepthDebugOverlay>();
            _triplanarCache = FindAny<TriplanarCache>();
            _persistence = FindAny<RoomScanPersistence>();
            _session = FindAny<RoomScanSession>();
            _keyframeCollector = FindAny<KeyframeCollector>();
            _debugMenu = FindAny<DebugMenuController>();
            _inputHandler = FindAny<RoomScanInputHandler>();
            _roomAnchor = FindAny<RoomAnchorManager>();
            _eventSystem = FindAny<EventSystem>();
            _ovrInputModule = FindAny<OVRInputModule>();
            _vrRaycaster = FindAny<VRDocumentRaycaster>();
            _rayDriver = FindAny<ControllerRayDriver>();
            _panelInputConfig = FindAny<PanelInputConfiguration>();

            _textureRefinement = _roomScanner != null ? _roomScanner.GetComponent<TextureRefinement>() : null;

            _depthCaptureWired = _depthCapture != null && AreFieldsAssigned(_depthCapture,
                "depthNormalCompute", "depthDilationCompute", "bilateralFilterCompute");
            _volumeWired = _volumeIntegrator != null && AreFieldsAssigned(_volumeIntegrator,
                "compute");
            _meshMatWired = _meshExtractor != null && AreFieldsAssigned(_meshExtractor,
                "scanMeshMaterial");
            _triplanarWired = _triplanarCache != null && AreFieldsAssigned(_triplanarCache,
                "bakeCompute");
            _computeShaderWired = _meshExtractor != null && AreFieldsAssigned(_meshExtractor,
                "surfaceNetsCompute");
            _refinedShaderWired = _textureRefinement != null && AreFieldsAssigned(_textureRefinement,
                "refinedMeshShader", "refinedMeshBackfaceShader");
            _occlusionShaderWired = _textureRefinement != null && AreFieldsAssigned(_textureRefinement,
                "occlusionMeshShader");
            _atlasBakeComputeWired = _textureRefinement != null && AreFieldsAssigned(_textureRefinement,
                "atlasBakeCompute");
            _debugOverlayWired = _roomScanner != null && AreFieldsAssigned(_roomScanner,
                "debugOverlayShader");
            RefreshGSplat();
            RefreshAIDetection();
            RefreshVRProject();

            RefreshURPState();

            RefreshBuildingBlocksState();
            _boundarylessManifest = ManifestHasAllQuestVREntries();
            _cleartextAllowed = ManifestHasCleartextTraffic();
            _insecureHttpAllowed = PlayerSettings.insecureHttpOption != InsecureHttpOption.NotAllowed;
        }

        // Partial methods implemented in RoomScanSetupWizard.GSplat.cs when
        // HAS_GAUSSIAN_SPLATTING is defined; silent no-ops otherwise.
        partial void RefreshGSplat();
        partial void DrawGSplatOptionalStatus();
        partial void CheckGSplatAnyMissing(ref bool anyMissing);
        partial void DrawGSplatShaderStatus(ref bool needsFix);
        partial void WireGSplatComponents();
        partial void SetupGSplatIfAvailable(GameObject root);

        // Partial methods implemented in RoomScanSetupWizard.AIDetection.cs when
        // HAS_AI_INFERENCE is defined; silent no-ops otherwise.
        partial void RefreshAIDetection();
        partial void DrawAIDetectionOptionalStatus();
        partial void CheckAIDetectionAnyMissing(ref bool anyMissing);
        partial void DrawAIDetectionShaderStatus(ref bool needsFix);
        partial void WireAIDetectionComponents();
        partial void SetupAIDetectionIfAvailable(GameObject root);

        // Partial methods implemented in RoomScanSetupWizard.VRProject.cs.
        // Always present (no #if guard) because OpenXR + Meta XR are core deps.
        partial void RefreshVRProject();
        partial void DrawVRProjectSection();

        // =================================================================
        //  GUI
        // =================================================================

        void OnGUI()
        {
            DrawHeader();
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            GUILayout.Space(4);

            // Game-Ready Preset is the promoted, common workflow for game
            // developers — keep it at the top so it's the first thing seen.
            // Everything below is for inspection / piecemeal fixes / opt-in
            // modules / final "do absolutely everything" sweep.
            DrawGameReadyPreset();

            DrawPrerequisites();
            DrawProjectSettings();
            DrawComponents();
            DrawVRProjectSection();
            DrawShaderWiring();
            DrawNativePlugins();

            GUILayout.Space(12);
            DrawMasterButton();
            GUILayout.Space(8);

            EditorGUILayout.EndScrollView();
        }

        void DrawHeader()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            GUILayout.Label("Room Scan Setup", EditorStyles.boldLabel);
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Refresh", EditorStyles.toolbarButton, GUILayout.Width(60)))
                Refresh();
            EditorGUILayout.EndHorizontal();
        }

        // -- Prerequisites ------------------------------------------------

        void DrawPrerequisites()
        {
            BeginSection("PREREQUISITES");

            string urpLabel = _urpAssetCached != null
                ? $"URP pipeline asset wired ({_urpAssetCached.name})"
                : "URP pipeline asset wired";
            StatusRow(urpLabel, _urpConfigured);
            StatusRow("ARSession", _arSession != null);
            StatusRow("Camera Rig (OVRCameraRig / XROrigin)", _cameraRig != null);
            StatusRow("AROcclusionManager", _arOcclusion != null);

            if (!_urpConfigured)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Setup URP (Quest defaults)", GUILayout.Width(220)))
                {
                    EnsureURPSetup();
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (_arSession == null)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add ARSession", GUILayout.Width(200)))
                    FixARSession();
                EditorGUILayout.EndHorizontal();
            }

            if (_cameraRig == null)
            {
                EditorGUILayout.HelpBox(
                    "Add a Camera Rig via  Meta > Tools > Building Blocks.\n" +
                    "The wizard will add AROcclusionManager to it automatically.",
                    MessageType.Info);
            }
            else if (_arOcclusion == null)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add AROcclusionManager", GUILayout.Width(200)))
                    FixAROcclusion();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        void FixARSession()
        {
            var go = FindByName("AR Session");
            if (go == null)
            {
                go = new GameObject("AR Session");
                Undo.RegisterCreatedObjectUndo(go, "Create AR Session");
            }

            if (go.GetComponent<ARSession>() == null)
                Undo.AddComponent<ARSession>(go);

            MarkDirty();
            Refresh();
        }

        void FixAROcclusion()
        {
            if (_cameraRig == null) return;

            // Find the camera — typically CenterEyeAnchor or Camera child
            Camera cam = _cameraRig.GetComponentInChildren<Camera>();
            if (cam == null)
            {
                Debug.LogWarning("[RoomScan Setup] No Camera found under camera rig");
                return;
            }

            GameObject target = cam.gameObject;

            // Need ARCameraManager as well for AROcclusionManager to work.
            // Both throw a wall of "No active XRSubsystem" errors in Editor
            // play mode without an active XR loader (no Quest, no Quest
            // Link). On device they're fine. We previously tried to silence
            // the Editor errors with EditorPlayModeXRGuard but the AR
            // OnEnable order bug it relied on never reliably fired before
            // the AR components' own OnEnable, and the workaround
            // introduced its own NRE chain via AROcclusionManager.OnDisable
            // → DestroyTextures. Reverted; just live with the Editor errors
            // and build to device to actually test.
            if (target.GetComponent<ARCameraManager>() == null)
                Undo.AddComponent<ARCameraManager>(target);

            var occl = target.GetComponent<AROcclusionManager>();
            if (occl == null)
                occl = Undo.AddComponent<AROcclusionManager>(target);
            // Depth sensor starts only from DepthCapture.StartDepthCapture.
            occl.enabled = false;
            DisableIdleScanHardware();

            MarkDirty();
            Refresh();
        }

        /// <summary>
        /// Leave AROcclusionManager and PassthroughCameraAccess disabled in
        /// the scene. RoomScanner enables them for the scan window only.
        /// USE_SCENE / HEADSET_CAMERA permission is still requested by the host.
        /// </summary>
        void DisableIdleScanHardware()
        {
            foreach (var occl in Object.FindObjectsByType<AROcclusionManager>(FindObjectsInactive.Include))
            {
                if (occl == null || !occl.enabled) continue;
                occl.enabled = false;
                EditorUtility.SetDirty(occl);
            }
            foreach (var pca in Object.FindObjectsByType<PassthroughCameraAccess>(FindObjectsInactive.Include))
            {
                if (pca == null || !pca.enabled) continue;
                pca.enabled = false;
                EditorUtility.SetDirty(pca);
            }
        }

        // -- Project Settings ---------------------------------------------

        const string MANIFEST_PATH = "Assets/Plugins/Android/AndroidManifest.xml";

        // Every <uses-feature> + <uses-permission> entry that QRS or its
        // satellite modules expect at runtime. Some of these (HEADSET_CAMERA,
        // USE_SCENE, USE_ANCHOR_API, etc.) are NOT in Meta's templated
        // manifest set and get stripped any time OVRProjectSetup.FixAllAsync
        // or the Project Setup Tool regenerates the manifest from
        // OVRProjectConfig — hence this comprehensive ensure-pass that runs
        // AFTER Meta's tooling in the wizard orchestrators.
        //
        // Each entry is idempotent (skip if already present, never remove).
        struct ManifestFeature { public string Name; public bool Required; }
        static readonly ManifestFeature[] REQUIRED_FEATURES = new[]
        {
            new ManifestFeature { Name = "android.hardware.vr.headtracking", Required = true  },
            new ManifestFeature { Name = "oculus.software.handtracking",     Required = false },
            new ManifestFeature { Name = "com.oculus.feature.PASSTHROUGH",   Required = false },
            new ManifestFeature { Name = "com.oculus.feature.BOUNDARYLESS_APP", Required = true },
        };

        static readonly string[] REQUIRED_PERMISSIONS = new[]
        {
            "com.oculus.permission.HAND_TRACKING",
            "com.oculus.permission.USE_ANCHOR_API",
            "com.oculus.permission.USE_SCENE",
            "horizonos.permission.HEADSET_CAMERA",
        };

        // horizonos SDK declaration — anchored to a current floor so MR
        // features (camera, anchors) are exposed.
        const string HORIZONOS_NS = "http://schemas.horizonos/sdk";
        const string HORIZONOS_MIN_SDK_VERSION = "60";
        const string HORIZONOS_TARGET_SDK_VERSION = "85";

        void DrawProjectSettings()
        {
            BeginSection("PROJECT SETTINGS");

            StatusRow("AndroidManifest Quest VR entries (features + permissions)",
                      _boundarylessManifest);
            StatusRow("AndroidManifest cleartext HTTP (LAN)", _cleartextAllowed);
            StatusRow("Player Settings: Allow HTTP", _insecureHttpAllowed);

            if (!_boundarylessManifest)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add Quest VR Manifest Entries", GUILayout.Width(220)))
                {
                    EnsureQuestVRManifest();
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (!_cleartextAllowed)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Allow Cleartext HTTP", GUILayout.Width(200)))
                {
                    FixCleartextTraffic();
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (!_insecureHttpAllowed)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Allow HTTP in Player Settings", GUILayout.Width(200)))
                {
                    PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
                    Debug.Log("[RoomScan Setup] Set Player Settings > insecureHttpOption to AlwaysAllowed");
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        /// <summary>
        /// Returns true iff every entry in REQUIRED_FEATURES /
        /// REQUIRED_PERMISSIONS plus the horizonos SDK declaration is
        /// already present in the manifest. Used by the status row + by the
        /// orchestrators to decide whether to re-run the ensure-pass.
        /// </summary>
        static bool ManifestHasAllQuestVREntries()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);
            if (!File.Exists(fullPath)) return false;

            try
            {
                var doc = XDocument.Load(fullPath);
                if (doc.Root == null) return false;
                XNamespace android = "http://schemas.android.com/apk/res/android";
                XNamespace horizonos = HORIZONOS_NS;

                foreach (var f in REQUIRED_FEATURES)
                {
                    bool found = doc.Root.Elements("uses-feature")
                        .Any(e => e.Attribute(android + "name")?.Value == f.Name);
                    if (!found) return false;
                }

                foreach (var p in REQUIRED_PERMISSIONS)
                {
                    bool found = doc.Root.Elements("uses-permission")
                        .Any(e => e.Attribute(android + "name")?.Value == p);
                    if (!found) return false;
                }

                bool hasHorizonOsSdk = doc.Root.Elements(horizonos + "uses-horizonos-sdk").Any();
                if (!hasHorizonOsSdk) return false;

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Idempotent: adds every required uses-feature, uses-permission, and
        /// the horizonos:uses-horizonos-sdk declaration if any are missing.
        /// Never removes existing entries — safe to run after Meta's
        /// OVRProjectSetup.FixAllAsync has rewritten the manifest from
        /// OVRProjectConfig defaults (which strips MR-only permissions like
        /// HEADSET_CAMERA / USE_SCENE that aren't in OVR's template).
        /// </summary>
        static void EnsureQuestVRManifest()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);

            if (!File.Exists(fullPath))
            {
                EditorUtility.DisplayDialog("Room Scan Setup",
                    "AndroidManifest.xml not found at:\n" + MANIFEST_PATH + "\n\n" +
                    "Build the project once or create a custom manifest first.",
                    "OK");
                return;
            }

            try
            {
                var doc = XDocument.Load(fullPath);
                if (doc.Root == null)
                {
                    Debug.LogError("[RoomScan Setup] AndroidManifest.xml has no <manifest> root.");
                    return;
                }

                XNamespace android = "http://schemas.android.com/apk/res/android";
                XNamespace horizonos = HORIZONOS_NS;
                bool dirty = false;
                var added = new List<string>();

                // Make sure xmlns:horizonos is declared on root so the SDK
                // element below can use the prefix without serializing as
                // xmlns="...". Unity templates usually include it but
                // OVR-regenerated manifests may not.
                if (doc.Root.Attribute(XNamespace.Xmlns + "horizonos") == null)
                {
                    doc.Root.Add(new XAttribute(XNamespace.Xmlns + "horizonos", HORIZONOS_NS));
                    dirty = true;
                }

                // <uses-feature>
                foreach (var f in REQUIRED_FEATURES)
                {
                    bool exists = doc.Root.Elements("uses-feature")
                        .Any(e => e.Attribute(android + "name")?.Value == f.Name);
                    if (exists) continue;

                    var el = new XElement("uses-feature",
                        new XAttribute(android + "name", f.Name),
                        new XAttribute(android + "required", f.Required ? "true" : "false"));

                    // headtracking gets a version attr by Android convention.
                    if (f.Name == "android.hardware.vr.headtracking")
                        el.Add(new XAttribute(android + "version", "1"));

                    doc.Root.Add(el);
                    added.Add($"feature:{f.Name}");
                    dirty = true;
                }

                // <uses-permission>
                foreach (var p in REQUIRED_PERMISSIONS)
                {
                    bool exists = doc.Root.Elements("uses-permission")
                        .Any(e => e.Attribute(android + "name")?.Value == p);
                    if (exists) continue;

                    doc.Root.Add(new XElement("uses-permission",
                        new XAttribute(android + "name", p)));
                    added.Add($"perm:{p}");
                    dirty = true;
                }

                // <horizonos:uses-horizonos-sdk>
                bool hasHorizonOsSdk = doc.Root.Elements(horizonos + "uses-horizonos-sdk").Any();
                if (!hasHorizonOsSdk)
                {
                    doc.Root.Add(new XElement(horizonos + "uses-horizonos-sdk",
                        new XAttribute(horizonos + "minSdkVersion",    HORIZONOS_MIN_SDK_VERSION),
                        new XAttribute(horizonos + "targetSdkVersion", HORIZONOS_TARGET_SDK_VERSION)));
                    added.Add("horizonos:uses-horizonos-sdk");
                    dirty = true;
                }

                if (!dirty)
                {
                    return;
                }

                doc.Save(fullPath);
                AssetDatabase.Refresh();
                Debug.Log($"[RoomScan Setup] AndroidManifest: added {added.Count} entr{(added.Count == 1 ? "y" : "ies")} \u2192 " +
                          string.Join(", ", added));
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] Failed to update manifest: {ex.Message}\n{ex.StackTrace}");
            }
        }

        // -- Cleartext HTTP -----------------------------------------------

        const string ANDROIDLIB_DIR = "Assets/Plugins/Android/NetworkSecurityConfig.androidlib";
        const string ANDROIDLIB_NSC = ANDROIDLIB_DIR + "/res/xml/network_security_config.xml";

        static bool ManifestHasCleartextTraffic()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);
            if (!File.Exists(fullPath)) return false;

            try
            {
                var doc = XDocument.Load(fullPath);
                XNamespace android = "http://schemas.android.com/apk/res/android";
                var app = doc.Root?.Element("application");
                if (app == null) return false;

                string val = app.Attribute(android + "usesCleartextTraffic")?.Value;
                if (val != "true") return false;

                string nscFull = Path.Combine(Application.dataPath, "..", ANDROIDLIB_NSC);
                return File.Exists(nscFull);
            }
            catch
            {
                return false;
            }
        }

        static void FixCleartextTraffic()
        {
            string manifestFull = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);

            if (!File.Exists(manifestFull))
            {
                EditorUtility.DisplayDialog("Room Scan Setup",
                    "AndroidManifest.xml not found at:\n" + MANIFEST_PATH + "\n\n" +
                    "Build the project once or create a custom manifest first.",
                    "OK");
                return;
            }

            try
            {
                var doc = XDocument.Load(manifestFull);
                XNamespace android = "http://schemas.android.com/apk/res/android";
                var app = doc.Root?.Element("application");
                if (app == null) return;

                // android:usesCleartextTraffic="true"
                var cleartext = app.Attribute(android + "usesCleartextTraffic");
                if (cleartext == null)
                    app.Add(new XAttribute(android + "usesCleartextTraffic", "true"));
                else
                    cleartext.Value = "true";

                // android:networkSecurityConfig="@xml/network_security_config"
                var nscAttr = app.Attribute(android + "networkSecurityConfig");
                if (nscAttr == null)
                    app.Add(new XAttribute(android + "networkSecurityConfig", "@xml/network_security_config"));
                else
                    nscAttr.Value = "@xml/network_security_config";

                doc.Save(manifestFull);
                Debug.Log("[RoomScan Setup] Added cleartext HTTP attributes to AndroidManifest.xml");

                // Unity 6+ requires Android resources in an .androidlib, not raw res/
                string libRoot = Path.Combine(Application.dataPath, "..", ANDROIDLIB_DIR);
                string nscDir = Path.Combine(libRoot, "res", "xml");
                if (!Directory.Exists(nscDir))
                    Directory.CreateDirectory(nscDir);

                string nscFull = Path.Combine(nscDir, "network_security_config.xml");
                if (!File.Exists(nscFull))
                {
                    const string nscContent =
                        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
                        "<network-security-config>\n" +
                        "    <base-config cleartextTrafficPermitted=\"true\">\n" +
                        "        <trust-anchors>\n" +
                        "            <certificates src=\"system\" />\n" +
                        "        </trust-anchors>\n" +
                        "    </base-config>\n" +
                        "</network-security-config>\n";
                    File.WriteAllText(nscFull, nscContent);
                }

                // AndroidManifest.xml for the library module
                string libManifest = Path.Combine(libRoot, "AndroidManifest.xml");
                if (!File.Exists(libManifest))
                {
                    const string libManifestContent =
                        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
                        "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\"\n" +
                        "    package=\"com.genesis.roomscan.netsecconfig\">\n" +
                        "</manifest>\n";
                    File.WriteAllText(libManifest, libManifestContent);
                }

                // project.properties marks it as a library
                string projProps = Path.Combine(libRoot, "project.properties");
                if (!File.Exists(projProps))
                    File.WriteAllText(projProps, "android.library=true\n");

                Debug.Log($"[RoomScan Setup] Created {ANDROIDLIB_DIR} with network_security_config.xml");
                AssetDatabase.Refresh();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] Failed to enable cleartext traffic: {ex.Message}");
            }
        }

        // -- Components ---------------------------------------------------

        void DrawComponents()
        {
            // ── Core (required) ──
            BeginSection("CORE COMPONENTS (Required)");

            StatusRow("RoomScanner", _roomScanner != null);
            StatusRow("DepthCapture", _depthCapture != null);
            StatusRow("VolumeIntegrator", _volumeIntegrator != null);
            StatusRow("MeshExtractor", _meshExtractor != null);
            StatusRow("RoomScanPersistence", _persistence != null);
            StatusRow("RoomAnchorManager (MRUK + SpatialAnchor)", _roomAnchor != null);

            var ovrConfig = OVRProjectConfig.CachedProjectConfig;
            bool anchorSupportOk = ovrConfig != null
                && ovrConfig.anchorSupport != OVRProjectConfig.AnchorSupport.Disabled;
            StatusRow("OVRProjectConfig anchor support", anchorSupportOk);
            if (!anchorSupportOk && ovrConfig != null)
            {
                if (GUILayout.Button("Fix: Enable Spatial Anchor Support"))
                {
                    ovrConfig.anchorSupport = OVRProjectConfig.AnchorSupport.Enabled;
                    OVRProjectConfig.CommitProjectConfig(ovrConfig);
                }
            }

            bool coreMissing = _roomScanner == null || _depthCapture == null ||
                               _volumeIntegrator == null || _meshExtractor == null ||
                               _persistence == null || _roomAnchor == null;
            if (coreMissing)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add Core Components", GUILayout.Width(180)))
                    FixCoreComponents();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();

            // ── Optional modules ──
            BeginSection("OPTIONAL MODULES");
            EditorGUILayout.HelpBox(
                "These are optional. Add them via the RoomScanner inspector's \"Add Module\" dropdown, or use \"Add All\" below.",
                MessageType.Info);

            StatusRowOptional("PassthroughCameraProvider", _cameraProvider != null);
            StatusRowOptional("PassthroughCameraAccess", _pcaComponent != null);
            StatusRowOptional("TriplanarCache", _triplanarCache != null);
            StatusRowOptional("KeyframeCollector", _keyframeCollector != null);
            DrawGSplatOptionalStatus();
            DrawAIDetectionOptionalStatus();
            StatusRowOptional("TextureRefinement", _roomScanner != null && _roomScanner.GetComponent<TextureRefinement>() != null);
            StatusRowOptional("RoomUnderstanding (MRUK bridge)", _roomScanner != null && _roomScanner.GetComponent<RoomUnderstanding>() != null);
            StatusRowOptional("CameraDebugOverlay", _cameraDebug != null);
            StatusRowOptional("DepthDebugOverlay", _depthDebug != null);
            StatusRowOptional("RoomScanInputHandler", _inputHandler != null);
            StatusRowOptional("DebugMenuController (HUD)", _debugMenu != null);

            bool anyOptionalMissing = _cameraProvider == null || _triplanarCache == null ||
                                      _debugMenu == null ||
                                      _inputHandler == null;
            CheckGSplatAnyMissing(ref anyOptionalMissing);
            CheckAIDetectionAnyMissing(ref anyOptionalMissing);
            if (anyOptionalMissing)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add All Optional", GUILayout.Width(160)))
                    FixAllOptionalModules();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();

            // Game-Ready Preset is rendered at the very top of the wizard
            // (see OnGUI) — it is the promoted workflow.

            // ── Debug Preset ──
            DrawDebugPreset();
        }

        GameObject FindOrCreateRoot()
        {
            GameObject root = null;
            if (_roomScanner != null)
                root = _roomScanner.gameObject;
            else if (_depthCapture != null)
                root = _depthCapture.gameObject;

            if (root == null)
            {
                root = FindByName("RoomScan");
                if (root == null)
                {
                    root = new GameObject("RoomScan");
                    Undo.RegisterCreatedObjectUndo(root, "Create RoomScan");
                }
            }

            EnsureRoomScanIdentityTransform(root);
            return root;
        }

        // The RoomScan GameObject MUST have an identity local transform.
        // RoomScanner spawns a child "RefinedMeshRenderer" whose mesh
        // vertices are pre-baked into world-space coordinates by
        // RoomScanPersistence.RelocateVertices, and RefinedMesh.shader
        // bypasses the object-to-world matrix (uses posWS directly). Unity
        // still uses localToWorldMatrix for frustum-cull bounds, so any
        // non-identity scale on this GameObject silently shrinks (or moves)
        // the culling box away from the actual geometry — the rendered mesh
        // then "disappears" from most viewing angles unless the camera
        // frustum happens to clip the displaced bounds. Reset defensively.
        static void EnsureRoomScanIdentityTransform(GameObject root)
        {
            if (root == null) return;
            var t = root.transform;

            bool wrongScale = t.localScale != Vector3.one;
            bool wrongPos = t.localPosition != Vector3.zero;
            bool wrongRot = t.localRotation != Quaternion.identity;
            if (!wrongScale && !wrongPos && !wrongRot) return;

            Undo.RecordObject(t, "Reset RoomScan transform to identity");
            t.localScale = Vector3.one;
            t.localPosition = Vector3.zero;
            t.localRotation = Quaternion.identity;
            EditorUtility.SetDirty(root);

            Debug.LogWarning(
                $"[RoomScanSetupWizard] Reset '{root.name}' transform to identity " +
                $"(wrongScale={wrongScale}, wrongPos={wrongPos}, wrongRot={wrongRot}). " +
                "RoomScanner's refined-mesh renderer hosts pre-baked world-space " +
                "vertices and bypasses object-to-world in its shader; any non-identity " +
                "parent transform breaks frustum-cull bounds and the mesh disappears " +
                "from most viewing angles. Don't put world-space UI panels (UIDocument) " +
                "directly on RoomScan — keep them on their own child GameObject.");
        }

        void FixCoreComponents()
        {
            var root = FindOrCreateRoot();

            // Adding RoomScanner auto-adds [RequireComponent] core siblings:
            // DepthCapture, VolumeIntegrator, MeshExtractor,
            // RoomScanPersistence, RoomAnchorManager
            if (root.GetComponent<RoomScanner>() == null)
                Undo.AddComponent<RoomScanner>(root);

            // Wire shader/compute on newly added core components
            foreach (var c in root.GetComponents<Component>())
                WireComponent(c);

            MarkDirty();
            Refresh();
        }

        void FixAllOptionalModules()
        {
            var root = FindOrCreateRoot();

            // Ensure core exists first
            if (root.GetComponent<RoomScanner>() == null)
                Undo.AddComponent<RoomScanner>(root);

            // PassthroughCameraAccess isn't pulled in by RequireComponent.
            // It will spam "No active XRSubsystem" / NRE errors in Editor
            // play mode without an XR loader; that's expected and can't be
            // fixed from outside Meta's package — build to device to test.
            if (root.GetComponent<PassthroughCameraAccess>() == null)
            {
                var pca = Undo.AddComponent<PassthroughCameraAccess>(root);
                pca.enabled = false;
            }
            if (root.GetComponent<PassthroughCameraProvider>() == null)
                Undo.AddComponent<PassthroughCameraProvider>(root);

            if (root.GetComponent<TriplanarCache>() == null)
                Undo.AddComponent<TriplanarCache>(root);
            if (root.GetComponent<TextureRefinement>() == null)
                Undo.AddComponent<TextureRefinement>(root);
            if (root.GetComponent<RoomUnderstanding>() == null)
                Undo.AddComponent<RoomUnderstanding>(root);

            // Public game-dev facade — see comment in AddGameReadyComponentsToRoot.
            if (root.GetComponent<RoomScanSession>() == null)
                Undo.AddComponent<RoomScanSession>(root);

            SetupGSplatIfAvailable(root);
            SetupAIDetectionIfAvailable(root);

            // Optional components not covered by RequireComponent
            if (root.GetComponent<RoomScanInputHandler>() == null)
                Undo.AddComponent<RoomScanInputHandler>(root);

            // Debug overlays — disabled by default
            if (root.GetComponent<CameraDebugOverlay>() == null)
            {
                var c = Undo.AddComponent<CameraDebugOverlay>(root);
                c.enabled = false;
            }
            if (root.GetComponent<DepthDebugOverlay>() == null)
            {
                var c = Undo.AddComponent<DepthDebugOverlay>(root);
                c.enabled = false;
            }

            // DebugMenu lives on a child (needs UIDocument)
            if (FindAny<DebugMenuController>() == null)
            {
                var debugGo = new GameObject("DebugMenu");
                debugGo.transform.SetParent(root.transform);
                Undo.RegisterCreatedObjectUndo(debugGo, "Create DebugMenu");

                Undo.AddComponent<UIDocument>(debugGo);
                Undo.AddComponent<DebugMenuController>(debugGo);
            }

            // Always ensure UIDocument has its assets assigned
            EnsureDebugMenuAssets();

            // Wire all components (core + optional)
            foreach (var c in root.GetComponents<Component>())
                WireComponent(c);

            // EventSystem + VR controller UI input pipeline
            EnsureVRInputInfrastructure();

            DisableIdleScanHardware();

            MarkDirty();
            Refresh();
        }

        void FixComponents()
        {
            FixCoreComponents();
            FixAllOptionalModules();
        }

        // -- Game-Ready Preset ----------------------------------------------

        void DrawGameReadyPreset()
        {
            BeginSection("GAME-READY PRESET");
            EditorGUILayout.HelpBox(
                "One-click \"make this project actually buildable for Quest VR\":\n" +
                "  \u2022 Switch active build profile to Meta Quest if needed (re-click after the reload)\n" +
                "  \u2022 URP pipeline + renderer at Assets/Settings/ with Quest-friendly defaults (4x MSAA, no HDR, single shadow cascade)\n" +
                "  \u2022 VR project prerequisites (XR Plug-in, OpenXR features, OVRProjectConfig \u2014 Outstanding tier)\n" +
                "  \u2022 AndroidManifest: full Quest VR feature/permission set (HEADSET_CAMERA, USE_SCENE, USE_ANCHOR_API, BOUNDARYLESS, etc.) + cleartext HTTP + insecureHttpOption\n" +
                "  \u2022 Meta XR Building Blocks: OVRCameraRig, Passthrough Underlay, PassthroughCameraAccess\n" +
                "  \u2022 AR Session + AROcclusionManager on the camera rig\n" +
                "  \u2022 Game-ready scene modules (scan \u2192 refine \u2192 release GPU \u2192 play)\n" +
                "  \u2022 Shader wiring + xatlas native plugin build (background)\n" +
                "Skips TriplanarCache, Gaussian Splat, and debug tools to keep the build lean.",
                MessageType.Info);

            // ── Scene-level state ──
            bool hasPCA = _pcaComponent != null;
            bool hasPCAProvider = _cameraProvider != null;
            bool hasRefinement = _textureRefinement != null;
            bool hasRoomUnderstanding = _roomScanner != null && _roomScanner.GetComponent<RoomUnderstanding>() != null;

            StatusRowOptional("PassthroughCameraAccess (camera RGB)", hasPCA);
            StatusRowOptional("PassthroughCameraProvider", hasPCAProvider);
            StatusRowOptional("TextureRefinement (atlas baking)", hasRefinement);
            StatusRowOptional("RoomUnderstanding (MRUK bridge)", hasRoomUnderstanding);
            StatusRowOptional("RoomScanSession (game-dev async API: StartScanAsync / UnloadActiveScanAsync / FinalizeScanAsync / LoadAsync)",
                              _session != null);

            if (hasRefinement)
            {
                var so = new SerializedObject(_textureRefinement);
                var simplifyProp = so.FindProperty("postBakeSimplificationRatio");
                if (simplifyProp != null)
                {
                    float val = simplifyProp.floatValue;
                    bool configured = val < 1f;
                    StatusRowOptional($"Mesh simplification ({val:P0})", configured);
                }
            }

            // ── Project-level state (also fixed by this preset) ──
            GUILayout.Space(2);
            EditorGUILayout.LabelField("Project prerequisites", EditorStyles.miniLabel);
            bool buildTargetIsAndroid = EditorUserBuildSettings.activeBuildTarget == BuildTarget.Android;
            bool activeProfileIsMetaQuest = IsActiveProfileMetaQuest();
            string profileLabel = activeProfileIsMetaQuest
                ? "Meta Quest"
                : (buildTargetIsAndroid ? "Android (plain)" : EditorUserBuildSettings.activeBuildTarget.ToString());
            StatusRowOptional($"Active build profile = Meta Quest (current: {profileLabel})", activeProfileIsMetaQuest);
            StatusRowOptional("URP pipeline asset (Quest defaults)", _urpConfigured);
            StatusRowOptional("Meta XR Building Blocks (Camera Rig + Passthrough + PCA)", _bbAllPresent);
            StatusRowOptional("Passthrough scene config (OVRManager + transparent center camera; no startup permission dialog)",
                              _ovrPassthroughReady);
            StatusRowOptional("AR Session + AROcclusionManager", _arSession != null && _arOcclusion != null);
            StatusRowOptional("AndroidManifest (Quest VR features + permissions + cleartext)",
                              _boundarylessManifest && _cleartextAllowed);
            StatusRowOptional("Player Settings: Allow HTTP", _insecureHttpAllowed);
            StatusRowOptional($"VR Project Bootstrap ({_vrOutstanding.Count} outstanding)", _vrOutstanding.Count == 0);
            StatusRowOptional("xatlas native plugins (Android + Editor)", _xatlasAndroid && _xatlasEditor);

            bool triplanarAttached = _triplanarCache != null;
            if (triplanarAttached)
            {
                var prev = GUI.color;
                GUI.color = COL_WARN;
                EditorGUILayout.BeginHorizontal();
                GUILayout.Space(12);
                GUILayout.Label("\u26A0", EditorStyles.boldLabel, GUILayout.Width(18));
                GUI.color = prev;
                GUILayout.Label("TriplanarCache is attached (\u2212240 MB GPU if removed)", GUILayout.ExpandWidth(true));
                prev = GUI.color;
                GUI.color = COL_WARN;
                GUILayout.Label("Optional", EditorStyles.miniLabel, GUILayout.Width(60));
                GUI.color = prev;
                EditorGUILayout.EndHorizontal();
            }

            bool sceneMissing   = !hasPCA || !hasPCAProvider || !hasRefinement || !hasRoomUnderstanding
                                  || _session == null;
            bool projectMissing = !buildTargetIsAndroid
                                  || !activeProfileIsMetaQuest
                                  || !_urpConfigured
                                  || !_bbAllPresent
                                  || !_ovrPassthroughReady
                                  || _arSession == null || _arOcclusion == null
                                  || !_boundarylessManifest || !_cleartextAllowed || !_insecureHttpAllowed
                                  || _vrOutstanding.Count > 0
                                  || !_xatlasAndroid || !_xatlasEditor;

            if (sceneMissing || projectMissing)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                using (new EditorGUI.DisabledScope(_gameReadyFixInProgress))
                {
                    if (GUILayout.Button("Apply Game-Ready Setup", GUILayout.Width(220)))
                        FixGameReadyModules();
                }
                EditorGUILayout.EndHorizontal();

                if (_gameReadyFixInProgress)
                    EditorGUILayout.HelpBox("Game-Ready setup in progress (VR bootstrap + Meta XR sweep)\u2026", MessageType.Info);
            }

            EndSection();
        }

        void DrawDebugPreset()
        {
            BeginSection("DEBUG PRESET");
            EditorGUILayout.HelpBox(
                "Development tools: debug HUD, input handler, camera/depth overlays, " +
                "and VR input pipeline for interacting with the debug menu. " +
                "Overlays are added disabled by default.",
                MessageType.Info);

            bool hasInput = _inputHandler != null;
            bool hasDebug = _debugMenu != null;
            bool hasCamOverlay = _cameraDebug != null;
            bool hasDepthOverlay = _depthDebug != null;

            StatusRowOptional("RoomScanInputHandler (VR controls)", hasInput);
            StatusRowOptional("DebugMenuController (HUD)", hasDebug);
            StatusRowOptional("CameraDebugOverlay (disabled)", hasCamOverlay);
            StatusRowOptional("DepthDebugOverlay (disabled)", hasDepthOverlay);

            GUILayout.Space(4);
            EditorGUILayout.LabelField("VR Input (for debug menu buttons)", EditorStyles.miniLabel);
            StatusRowOptional("EventSystem + OVRInputModule", _eventSystem != null && _ovrInputModule != null);
            StatusRowOptional("VRDocumentRaycaster (UI pointer)", _vrRaycaster != null);
            StatusRowOptional("ControllerRayDriver (laser + cursor)", _rayDriver != null);
            StatusRowOptional("PanelInputConfiguration", _panelInputConfig != null);

            bool debugMissing = !hasInput || !hasDebug || !hasCamOverlay || !hasDepthOverlay
                                || _eventSystem == null || _ovrInputModule == null
                                || _vrRaycaster == null || _rayDriver == null
                                || _panelInputConfig == null;
            if (debugMissing)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add Debug Modules", GUILayout.Width(200)))
                    FixDebugModules();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        // Tracks whether either the Game-Ready preset or Setup Everything is
        // currently running, so the matching buttons can disable themselves and
        // we never re-enter the async orchestrator while VRProjectBootstrap.FixAllAsync
        // is still awaiting Meta's project setup tool.
        bool _gameReadyFixInProgress;

        async void FixGameReadyModules()
        {
            if (_gameReadyFixInProgress) return;
            _gameReadyFixInProgress = true;

            try
            {
                if (TrySwitchToAndroidBuildTarget("Game-Ready Setup")) return;

                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Auditing VR project settings\u2026", 0.05f);
                VRProjectBootstrap.Audit();

                // URP first — shaders fall back to magenta until the
                // pipeline asset exists and is wired into GraphicsSettings,
                // so any later step that touches a Material/Shader needs
                // this in place.
                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Ensuring URP pipeline + Quest-friendly defaults\u2026", 0.10f);
                EnsureURPSetup();

                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Fixing VR prerequisites (XR Plug-in, OpenXR, OVRProjectConfig\u2026)", 0.15f);
                await VRProjectBootstrap.FixAllAsync(CheckSeverity.Outstanding);

                // EnsureQuestVRManifest is unconditional (and idempotent) on
                // purpose — it has to undo any permission stripping that
                // OVRProjectSetup.FixAllAsync may have done a moment ago when
                // it regenerated the manifest from OVRProjectConfig defaults.
                // HEADSET_CAMERA / USE_SCENE / USE_ANCHOR_API are not in
                // Meta's templated set and would otherwise vanish here.
                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Updating AndroidManifest + Player Settings\u2026", 0.50f);
                EnsureQuestVRManifest();
                if (!ManifestHasCleartextTraffic()) FixCleartextTraffic();
                if (PlayerSettings.insecureHttpOption == InsecureHttpOption.NotAllowed)
                {
                    PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
                    Debug.Log("[RoomScan Setup] Set Player Settings > insecureHttpOption to AlwaysAllowed");
                }

                // Meta XR Building Blocks: drops in OVRCameraRig +
                // Passthrough Underlay + PassthroughCameraAccess with
                // Meta's recommended wiring (TrackingOrigin = FloorLevel,
                // Underlay layer set up, etc.). Done before AR session
                // so AROcclusionManager can latch onto the new rig camera.
                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Installing Meta XR Building Blocks (Camera Rig + Passthrough)\u2026", 0.60f);
                await EnsureRequiredBuildingBlocksAsync();
                Refresh();

                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Setting up AR session + occlusion\u2026", 0.65f);
                if (_arSession == null) FixARSession();
                if (_cameraRig != null && _arOcclusion == null) FixAROcclusion();

                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Adding game-ready scene components\u2026", 0.80f);
                AddGameReadyComponentsToRoot();

                EditorUtility.DisplayProgressBar("Game-Ready Setup",
                    "Wiring shaders\u2026", 0.90f);
                FixShaderWiring();

                RefreshNativePlugins();
                bool xatlasMissing = !_xatlasAndroid || !_xatlasEditor;
                if (xatlasMissing)
                {
                    EditorUtility.DisplayProgressBar("Game-Ready Setup",
                        "Starting xatlas plugin build (background)\u2026", 0.97f);
                    BuildXAtlasPlugin();
                }

                Debug.Log("[RoomScan Setup] Game-Ready setup complete." +
                    (xatlasMissing ? " (xatlas build running in background)" : ""));
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] Game-Ready setup failed: {ex}");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                _gameReadyFixInProgress = false;
                MarkDirty();
                Refresh();
                Repaint();
            }
        }

        // Pure scene-component piece of the game-ready preset, callable
        // synchronously from the orchestrator above without owning Refresh().
        void AddGameReadyComponentsToRoot()
        {
            var root = FindOrCreateRoot();

            if (root.GetComponent<RoomScanner>() == null)
                Undo.AddComponent<RoomScanner>(root);

            // PassthroughCameraAccess is normally added by the Meta XR
            // Building Block (see EnsureRequiredBuildingBlocksAsync), but
            // fall back to a root-level component if the block didn't land
            // anywhere in the scene — RoomScanner needs a PCA somewhere.
            // PCA + ARSession + AROcclusionManager will spam errors in
            // Editor play mode without an XR loader; that's expected,
            // build to device.
            if (UnityEngine.Object.FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include) == null)
            {
                var pca = Undo.AddComponent<PassthroughCameraAccess>(root);
                pca.enabled = false;
            }
            if (root.GetComponent<PassthroughCameraProvider>() == null)
                Undo.AddComponent<PassthroughCameraProvider>(root);

            if (root.GetComponent<TextureRefinement>() == null)
                Undo.AddComponent<TextureRefinement>(root);
            if (root.GetComponent<RoomUnderstanding>() == null)
                Undo.AddComponent<RoomUnderstanding>(root);

            // RoomScanSession: public game-dev facade (StartScanAsync / UnloadActiveScanAsync / FinalizeScanAsync /
            // LoadLatestAsync / HasSavedScan / ProgressUpdated). Without it,
            // game code that follows the documented public-API path cannot
            // find RoomScanSession.Instance and bails.
            if (root.GetComponent<RoomScanSession>() == null)
                Undo.AddComponent<RoomScanSession>(root);

            ApplyGameReadyScanDefaults(root);

            foreach (var c in root.GetComponents<Component>())
                WireComponent(c);

            DisableIdleScanHardware();
        }

        /// <summary>
        /// Stamp the game-proven scan knobs onto an existing RoomScan root
        /// (extract 8 Hz, throttled keyframes, 50% post-bake simplify).
        /// Confine-to-room is host art direction and is left alone.
        /// New components already get these from field defaults; this
        /// covers scenes serialized against older 30 Hz / burst-keyframe
        /// values.
        /// </summary>
        void ApplyGameReadyScanDefaults(GameObject root)
        {
            var scanner = root.GetComponent<RoomScanner>();
            if (scanner != null)
            {
                var so = new SerializedObject(scanner);
                var hz = so.FindProperty("meshExtractionHz");
                if (hz != null) hz.floatValue = 8f;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(scanner);
            }

            var kf = root.GetComponent<KeyframeCollector>();
            if (kf != null)
            {
                var so = new SerializedObject(kf);
                var move = so.FindProperty("moveThreshold");
                if (move != null) move.floatValue = 0.15f;
                var rot = so.FindProperty("rotateThresholdDeg");
                if (rot != null) rot.floatValue = 10f;
                var interval = so.FindProperty("minCaptureInterval");
                if (interval != null) interval.floatValue = 0.25f;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(kf);
            }

            var tr = root.GetComponent<TextureRefinement>();
            if (tr != null)
            {
                var so = new SerializedObject(tr);
                var simplifyProp = so.FindProperty("postBakeSimplificationRatio");
                if (simplifyProp != null && simplifyProp.floatValue >= 1f)
                {
                    simplifyProp.floatValue = 0.5f;
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(tr);
                    Debug.Log("[RoomScan Setup] Set postBakeSimplificationRatio to 0.5 for game-ready mesh");
                }
            }
        }

        /// <summary>
        /// If the active build target is not Android, switches it (which
        /// triggers a domain reload and aborts the current async pipeline)
        /// and returns true so the caller bails out cleanly. The user is
        /// informed via dialog that they need to re-click after the reload.
        /// </summary>
        bool TrySwitchToAndroidBuildTarget(string flowName)
        {
            // Already on the Meta Quest profile? Nothing to do.
            if (EditorUserBuildSettings.activeBuildTarget == BuildTarget.Android
                && IsActiveProfileMetaQuest())
                return false;

            EditorUtility.ClearProgressBar();

            // Try the Meta Quest *build profile* first (Unity 6.1+). It's a
            // derived Android profile that ships Quest-tuned Player + Quality
            // overrides (Vulkan, IL2CPP, ARM64, Multiview, Quest quality
            // level), and it's what the user picks by hand in File > Build
            // Profiles. Falls back to plain Android if Meta Quest isn't
            // registered (older Unity, missing Android module, etc.).
            string what = "Meta Quest build profile";
            EditorUtility.DisplayDialog(flowName,
                "Active build target is " + EditorUserBuildSettings.activeBuildTarget +
                (IsActiveProfileMetaQuest() ? " (Meta Quest profile)" : "") + ".\n\n" +
                "Switching to the " + what + " now \u2014 this triggers a domain reload " +
                "and aborts the rest of this run.\n\n" +
                "Click \"" + flowName + "\" again after Unity finishes reloading to " +
                "apply the remaining fixes.",
                "Switch and reload");

            if (!TryActivateMetaQuestProfile())
            {
                Debug.LogWarning("[RoomScan Setup] Meta Quest classic build profile not " +
                                 "found \u2014 falling back to plain Android target. " +
                                 "Run File > Build Profiles once to let Unity register the " +
                                 "Meta Quest platform, then re-run this wizard.");
                EditorUserBuildSettings.SwitchActiveBuildTarget(
                    BuildTargetGroup.Android, BuildTarget.Android);
            }

            // Drop the in-progress flag — the domain reload will wipe state
            // anyway, but if for some reason it doesn't fire we don't want
            // to leave the wizard locked out forever.
            _gameReadyFixInProgress = false;
            return true;
        }

        void FixDebugModules()
        {
            var root = FindOrCreateRoot();

            if (root.GetComponent<RoomScanner>() == null)
                Undo.AddComponent<RoomScanner>(root);

            if (root.GetComponent<RoomScanInputHandler>() == null)
                Undo.AddComponent<RoomScanInputHandler>(root);

            if (root.GetComponent<CameraDebugOverlay>() == null)
            {
                var c = Undo.AddComponent<CameraDebugOverlay>(root);
                c.enabled = false;
            }
            if (root.GetComponent<DepthDebugOverlay>() == null)
            {
                var c = Undo.AddComponent<DepthDebugOverlay>(root);
                c.enabled = false;
            }

            if (FindAny<DebugMenuController>() == null)
            {
                var debugGo = new GameObject("DebugMenu");
                debugGo.transform.SetParent(root.transform);
                Undo.RegisterCreatedObjectUndo(debugGo, "Create DebugMenu");
                Undo.AddComponent<UIDocument>(debugGo);
                Undo.AddComponent<DebugMenuController>(debugGo);
            }
            EnsureDebugMenuAssets();

            foreach (var c in root.GetComponents<Component>())
                WireComponent(c);

            EnsureVRInputInfrastructure();

            MarkDirty();
            Refresh();
        }

        /// <summary>
        /// Static entry point for ensuring VR input infrastructure exists.
        /// Called by <see cref="RoomScannerEditor"/> when adding the Debug Menu module.
        /// </summary>
        internal static void EnsureVRInput()
        {
            // EventSystem
            var es = FindAny<EventSystem>();
            if (es == null)
            {
                var esGo = new GameObject("EventSystem");
                Undo.RegisterCreatedObjectUndo(esGo, "Create EventSystem");
                es = Undo.AddComponent<EventSystem>(esGo);
            }

            if (es.GetComponent<OVRInputModule>() == null)
            {
                var standalone = es.GetComponent<StandaloneInputModule>();
                if (standalone != null) Undo.DestroyObjectImmediate(standalone);
                Undo.AddComponent<OVRInputModule>(es.gameObject);
            }

            if (es.GetComponent<PanelInputConfiguration>() == null)
            {
                var pic = Undo.AddComponent<PanelInputConfiguration>(es.gameObject);
                var so = new SerializedObject(pic);
                SetBool(so, "m_DefaultEventCameraIsMainCamera", true);
                SetBool(so, "m_AutoCreatePanelComponents", true);
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(pic);
            }

            if (es.GetComponent<VRDocumentRaycaster>() == null)
                Undo.AddComponent<VRDocumentRaycaster>(es.gameObject);
            var rayDriver = es.GetComponent<ControllerRayDriver>();
            if (rayDriver == null)
                rayDriver = Undo.AddComponent<ControllerRayDriver>(es.gameObject);
            WireComponent(rayDriver);
        }

        void EnsureVRInputInfrastructure() => EnsureVRInput();

        static void SetBool(SerializedObject so, string fieldName, bool value)
        {
            var prop = so.FindProperty(fieldName);
            if (prop != null) prop.boolValue = value;
        }

        // -- Shader / Material Wiring ------------------------------------

        void DrawShaderWiring()
        {
            BeginSection("SHADER & MATERIAL WIRING");

            bool needsFix = false;

            // Core — always present
            if (_depthCapture != null)   { StatusRow("DepthCapture compute shaders", _depthCaptureWired); needsFix |= !_depthCaptureWired; }
            if (_volumeIntegrator != null){ StatusRow("VolumeIntegrator compute shader", _volumeWired);    needsFix |= !_volumeWired; }
            if (_meshExtractor != null)  { StatusRow("MeshExtractor scan material", _meshMatWired);        needsFix |= !_meshMatWired; }
            if (_meshExtractor != null)  { StatusRow("SurfaceNetsExtract compute shader", _computeShaderWired); needsFix |= !_computeShaderWired; }

            // Optional — only show if the module is attached
            if (_triplanarCache != null) { StatusRow("TriplanarCache bake compute", _triplanarWired);      needsFix |= !_triplanarWired; }

            if (_textureRefinement != null)   { StatusRow("RefinedMesh shader (texture refine)", _refinedShaderWired); needsFix |= !_refinedShaderWired; }
            if (_textureRefinement != null)   { StatusRow("OcclusionMesh shader (MR occluder)", _occlusionShaderWired); needsFix |= !_occlusionShaderWired; }
            if (_textureRefinement != null)   { StatusRow("AtlasBakeCompute (GPU bake)", _atlasBakeComputeWired);      needsFix |= !_atlasBakeComputeWired; }
            if (_roomScanner != null)        { StatusRow("DebugOverlay shader (scene viz)", _debugOverlayWired);     needsFix |= !_debugOverlayWired; }
            DrawGSplatShaderStatus(ref needsFix);
            DrawAIDetectionShaderStatus(ref needsFix);

            if (needsFix)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Wire All Shaders", GUILayout.Width(160)))
                    FixShaderWiring();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        void FixShaderWiring()
        {
            WireComponent(_depthCapture);
            WireComponent(_volumeIntegrator);
            WireComponent(_meshExtractor);
            WireComponent(_triplanarCache);
            WireComponent(_depthDebug);
            WireComponent(_roomScanner);

            var tr = _roomScanner != null ? _roomScanner.GetComponent<TextureRefinement>() : null;
            WireComponent(tr);
            WireGSplatComponents();
            WireAIDetectionComponents();

            var rayDriver = FindAny<UI.ControllerRayDriver>();
            WireComponent(rayDriver);

            MarkDirty();
            Refresh();
        }

        static void AssignCompute(SerializedObject so, string fieldName, string assetPath)
        {
            AssignAsset<ComputeShader>(so, fieldName, assetPath);
        }

        static void AssignAsset<T>(SerializedObject so, string fieldName, string assetPath) where T : Object
        {
            var prop = so.FindProperty(fieldName);
            if (prop == null) return;
            if (prop.objectReferenceValue != null) return;

            var asset = AssetDatabase.LoadAssetAtPath<T>(assetPath);
            if (asset != null)
                prop.objectReferenceValue = asset;
            else
                Debug.LogWarning($"[RoomScan Setup] Could not find {assetPath}");
        }

        static Material GetOrCreateScanMaterial()
        {
            const string pkgMatPath = "Packages/com.genesis.roomscan/Runtime/Materials/ScanMesh.mat";
            var pkgMat = AssetDatabase.LoadAssetAtPath<Material>(pkgMatPath);
            if (pkgMat != null) return pkgMat;

            // Fallback: create in project if package material not found
            const string matPath = "Assets/RoomScan/ScanMesh.mat";
            var existing = AssetDatabase.LoadAssetAtPath<Material>(matPath);
            if (existing != null) return existing;

            Shader shader = Shader.Find("Genesis/ScanMeshVertexColor");
            if (shader == null)
            {
                Debug.LogWarning("[RoomScan Setup] Shader 'Genesis/ScanMeshVertexColor' not found");
                return null;
            }

            if (!AssetDatabase.IsValidFolder("Assets/RoomScan"))
                AssetDatabase.CreateFolder("Assets", "RoomScan");

            var mat = new Material(shader) { name = "ScanMesh", enableInstancing = true };
            AssetDatabase.CreateAsset(mat, matPath);
            AssetDatabase.SaveAssets();
            return mat;
        }

        // -- Native Plugins -----------------------------------------------

        bool _xatlasAndroid, _xatlasEditor;

        void RefreshNativePlugins()
        {
            string pkgRoot = "Packages/com.genesis.roomscan/Runtime";
            _xatlasAndroid = System.IO.File.Exists(
                Path.GetFullPath(Path.Combine(pkgRoot, "Plugins/Android/libxatlas.so")));
#if UNITY_EDITOR_WIN
            _xatlasEditor = System.IO.File.Exists(
                Path.GetFullPath(Path.Combine(pkgRoot, "Plugins/Windows/xatlas.dll")));
#elif UNITY_EDITOR_LINUX
            _xatlasEditor = System.IO.File.Exists(
                Path.GetFullPath(Path.Combine(pkgRoot, "Plugins/Linux/libxatlas.so")));
#else
            _xatlasEditor = System.IO.File.Exists(
                Path.GetFullPath(Path.Combine(pkgRoot, "Plugins/macOS/libxatlas.bundle")));
#endif
        }

        void DrawNativePlugins()
        {
            RefreshNativePlugins();
            BeginSection("NATIVE PLUGINS");
            StatusRow("xatlas (Android ARM64)", _xatlasAndroid);
#if UNITY_EDITOR_WIN
            StatusRow("xatlas (Windows Editor)", _xatlasEditor);
#elif UNITY_EDITOR_LINUX
            StatusRow("xatlas (Linux Editor)", _xatlasEditor);
#else
            StatusRow("xatlas (macOS Editor)", _xatlasEditor);
#endif

            if (!_xatlasAndroid || !_xatlasEditor)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Build xatlas Plugin", GUILayout.Width(200)))
                    BuildXAtlasPlugin();
                EditorGUILayout.EndHorizontal();
            }
            EndSection();
        }

        /// <summary>
        /// Wires shader/compute/material references on a freshly added component.
        /// Called by both the setup wizard and the RoomScannerEditor "Add Module" dropdown.
        /// </summary>
        internal static void WireComponent(Component component)
        {
            if (component == null) return;

            const string PKG_SHADERS = "Packages/com.genesis.roomscan/Runtime/Shaders/";

            switch (component)
            {
                case DepthCapture dc:
                {
                    var so = new SerializedObject(dc);
                    AssignCompute(so, "depthNormalCompute", PKG_SHADERS + "DepthNormals.compute");
                    AssignCompute(so, "depthDilationCompute", PKG_SHADERS + "DepthDilation.compute");
                    AssignCompute(so, "bilateralFilterCompute", PKG_SHADERS + "BilateralDepthFilter.compute");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(dc);
                    break;
                }
                case VolumeIntegrator vi:
                {
                    var so = new SerializedObject(vi);
                    AssignCompute(so, "compute", PKG_SHADERS + "VolumeIntegration.compute");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(vi);
                    break;
                }
                case MeshExtractor me:
                {
                    var so = new SerializedObject(me);
                    var prop = so.FindProperty("scanMeshMaterial");
                    if (prop != null && prop.objectReferenceValue == null)
                    {
                        Material mat = GetOrCreateScanMaterial();
                        if (mat != null) prop.objectReferenceValue = mat;
                    }
                    AssignCompute(so, "surfaceNetsCompute", PKG_SHADERS + "SurfaceNetsExtract.compute");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(me);
                    break;
                }
                case TriplanarCache tc:
                {
                    var so = new SerializedObject(tc);
                    AssignCompute(so, "bakeCompute", PKG_SHADERS + "TriplanarBake.compute");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(tc);
                    break;
                }
                case TextureRefinement tr:
                {
                    var so = new SerializedObject(tr);
                    AssignAsset<Shader>(so, "refinedMeshShader", PKG_SHADERS + "RefinedMesh.shader");
                    AssignAsset<Shader>(so, "refinedMeshBackfaceShader", PKG_SHADERS + "RefinedMeshBackface.shader");
                    AssignAsset<Shader>(so, "occlusionMeshShader", PKG_SHADERS + "OcclusionMesh.shader");
                    AssignAsset<ComputeShader>(so, "atlasBakeCompute", PKG_SHADERS + "AtlasBakeCompute.compute");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(tr);
                    break;
                }
                case DepthDebugOverlay dd:
                {
                    var so = new SerializedObject(dd);
                    AssignAsset<Shader>(so, "depthVisualizeShader", PKG_SHADERS + "DepthVisualize.shader");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(dd);
                    break;
                }
                case RoomScanner rs:
                {
                    var so = new SerializedObject(rs);
                    AssignAsset<Shader>(so, "debugOverlayShader", PKG_SHADERS + "DebugOverlay.shader");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(rs);
                    break;
                }
                case UI.ControllerRayDriver crd:
                {
                    var so = new SerializedObject(crd);
                    AssignAsset<Shader>(so, "overlayShader", PKG_SHADERS + "DebugOverlay.shader");
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(crd);
                    break;
                }
            }

#if HAS_GAUSSIAN_SPLATTING
            WireGSplatComponent(component);
#endif
#if HAS_AI_INFERENCE
            WireAIDetectionComponent(component);
#endif
        }

        internal static void BuildXAtlasPlugin()
        {
            string pkgRoot = Path.GetFullPath("Packages/com.genesis.roomscan/Runtime");
            string srcDir = Path.Combine(pkgRoot, "Native/xatlas");
            string srcApi = Path.Combine(srcDir, "xatlas_c_api.cpp");
            string srcImpl = Path.Combine(srcDir, "xatlas.cpp");
            string meshoptDir = Path.Combine(pkgRoot, "Native/meshoptimizer");
            string srcSimplifier = Path.Combine(meshoptDir, "simplifier.cpp");

            if (!System.IO.File.Exists(srcApi) || !System.IO.File.Exists(srcImpl))
            {
                EditorUtility.DisplayDialog("Build xatlas",
                    $"Source files not found in:\n{srcDir}\n\nExpected xatlas.cpp and xatlas_c_api.cpp",
                    "OK");
                return;
            }

            bool hasMeshopt = System.IO.File.Exists(srcSimplifier);
            if (!hasMeshopt)
                Debug.LogWarning("[RoomScan Setup] meshoptimizer sources not found — building without mesh simplification");

            string meshoptSrc = hasMeshopt ? $" \"{srcSimplifier}\"" : "";
            string meshoptInc = hasMeshopt ? $" -I\"{meshoptDir}\"" : "";

            var builds = new System.Collections.Generic.List<(string label, string exe, string args, string outAssetPath)>();

            // Host editor plugin (platform-specific)
#if UNITY_EDITOR_WIN
            {
                string clExe = FindMsvcCompiler();
                if (clExe != null)
                {
                    string outDir = Path.Combine(pkgRoot, "Plugins/Windows");
                    Directory.CreateDirectory(outDir);
                    string outPath = Path.Combine(outDir, "xatlas.dll");
                    string incFlags = hasMeshopt ? $" /I\"{meshoptDir}\"" : "";
                    string allSrc = $"\"{srcApi}\" \"{srcImpl}\"" + (hasMeshopt ? $" \"{srcSimplifier}\"" : "");
                    string bArgs = $"/nologo /O2 /std:c++14 /EHsc /LD{incFlags} {allSrc} /Fe:\"{outPath}\" /link /DLL";
                    builds.Add(("Windows xatlas", clExe, bArgs,
                        "Packages/com.genesis.roomscan/Runtime/Plugins/Windows/xatlas.dll"));
                }
                else
                {
                    string clangExe = FindHostClang();
                    if (clangExe != null)
                    {
                        string outDir = Path.Combine(pkgRoot, "Plugins/Windows");
                        Directory.CreateDirectory(outDir);
                        string outPath = Path.Combine(outDir, "xatlas.dll");
                        string bArgs = $"-shared -O2 -std=c++11{meshoptInc} " +
                                       $"-o \"{outPath}\" \"{srcApi}\" \"{srcImpl}\"{meshoptSrc}";
                        builds.Add(("Windows xatlas", clangExe, bArgs,
                            "Packages/com.genesis.roomscan/Runtime/Plugins/Windows/xatlas.dll"));
                    }
                    else
                    {
                        Debug.LogError("[RoomScan Setup] No C++ compiler found. Install Visual Studio " +
                            "with C++ Desktop workload, or add clang++/g++ to your PATH.");
                    }
                }
            }
#elif UNITY_EDITOR_LINUX
            {
                string outDir = Path.Combine(pkgRoot, "Plugins/Linux");
                Directory.CreateDirectory(outDir);
                string outPath = Path.Combine(outDir, "libxatlas.so");
                string bArgs = $"-shared -O2 -fPIC -std=c++11 -fvisibility=hidden{meshoptInc} " +
                               $"-o \"{outPath}\" \"{srcApi}\" \"{srcImpl}\"{meshoptSrc}";
                string compiler = FindHostClang() ?? "g++";
                builds.Add(("Linux xatlas", compiler, bArgs,
                    "Packages/com.genesis.roomscan/Runtime/Plugins/Linux/libxatlas.so"));
            }
#else // macOS
            {
                string outDir = Path.Combine(pkgRoot, "Plugins/macOS");
                Directory.CreateDirectory(outDir);
                string outPath = Path.Combine(outDir, "libxatlas.bundle");
                // Both editor architectures, against an SDK no newer than this macOS: the default SDK can be a newer
                // one the installed linker cannot read ("tapi error: unknown architecture" on its .tbd stubs).
                string bArgs = $"{MacSdkArgs()}-arch arm64 -arch x86_64 -shared -O2 -fPIC -std=c++11 -fvisibility=hidden{meshoptInc} " +
                               $"-o \"{outPath}\" \"{srcApi}\" \"{srcImpl}\"{meshoptSrc}";
                builds.Add(("macOS xatlas", "clang++", bArgs,
                    "Packages/com.genesis.roomscan/Runtime/Plugins/macOS/libxatlas.bundle"));
            }
#endif

            // Android ARM64 (cross-compile from any host)
            {
                string ndkClang = FindNdkClang();
                if (ndkClang != null)
                {
                    string outDir = Path.Combine(pkgRoot, "Plugins/Android");
                    Directory.CreateDirectory(outDir);
                    string outPath = Path.Combine(outDir, "libxatlas.so");
                    string bArgs = $"-shared -O2 -fPIC -std=c++11 -fvisibility=hidden{meshoptInc} " +
                                   $"-o \"{outPath}\" \"{srcApi}\" \"{srcImpl}\"{meshoptSrc}";
                    builds.Add(("Android xatlas", ndkClang, bArgs,
                        "Packages/com.genesis.roomscan/Runtime/Plugins/Android/libxatlas.so"));
                }
            }

            if (builds.Count == 0)
            {
                Debug.LogError("[RoomScan Setup] No build targets available");
                return;
            }

            StartAsyncBuilds(builds);
        }

        /// <summary>
        /// <c>-isysroot</c> for the newest installed macOS SDK whose version is not above the running macOS (Command Line
        /// Tools or Xcode), with a trailing space; empty when none is found (clang then uses its default SDK).
        /// </summary>
        static string MacSdkArgs()
        {
            try
            {
                var osv = RunTool("sw_vers", "-productVersion");
                if (!System.Version.TryParse(PadVersion(osv), out var os)) return "";
                string best = null;
                System.Version bestV = null;
                foreach (string root in new[]
                         {
                             "/Library/Developer/CommandLineTools/SDKs",
                             "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
                         })
                {
                    if (!Directory.Exists(root)) continue;
                    foreach (string dir in Directory.GetDirectories(root, "MacOSX*.sdk"))
                    {
                        string name = Path.GetFileNameWithoutExtension(dir).Substring("MacOSX".Length);
                        if (!System.Version.TryParse(PadVersion(name), out var v) || v > os) continue;
                        if (bestV == null || v > bestV) { bestV = v; best = dir; }
                    }
                }
                return best != null ? $"-isysroot \"{best}\" " : "";
            }
            catch
            {
                return "";
            }
        }

        static string PadVersion(string v)
        {
            v = (v ?? "").Trim();
            return v.Contains(".") ? v : v + ".0";
        }

        static string RunTool(string exe, string args)
        {
            var psi = new System.Diagnostics.ProcessStartInfo(exe, args)
            {
                RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true,
            };
            using var proc = System.Diagnostics.Process.Start(psi);
            string output = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(5000);
            return output;
        }

        static string FindNdkClang()
        {
            string ndkPath = null;
            try
            {
                ndkPath = UnityEditor.Android.AndroidExternalToolsSettings.ndkRootPath;
            }
            catch
            {
                Debug.LogWarning("[RoomScan Setup] Android NDK path not configured. Skipping Android build.");
                return null;
            }

            if (string.IsNullOrEmpty(ndkPath) || !Directory.Exists(ndkPath))
            {
                Debug.LogWarning($"[RoomScan Setup] NDK not found at: {ndkPath}");
                return null;
            }

            string prebuilt = Path.Combine(ndkPath, "toolchains/llvm/prebuilt");
            if (!Directory.Exists(prebuilt)) return null;

            string[] hosts = Directory.GetDirectories(prebuilt);
            if (hosts.Length == 0) return null;

            string binDir = Path.Combine(hosts[0], "bin");
            // Windows NDK ships .cmd wrappers; Unix has bare executables
            string[] candidates = {
                Path.Combine(binDir, "aarch64-linux-android31-clang++.cmd"),
                Path.Combine(binDir, "aarch64-linux-android31-clang++.exe"),
                Path.Combine(binDir, "aarch64-linux-android31-clang++"),
            };
            foreach (string c in candidates)
                if (System.IO.File.Exists(c)) return c;

            Debug.LogWarning($"[RoomScan Setup] NDK clang++ not found in {binDir}");
            return null;
        }

        static string FindHostClang()
        {
            // Check common locations for clang++ on the host
            string[] candidates;
#if UNITY_EDITOR_WIN
            candidates = new[] { "clang++.exe", "clang++", "g++.exe" };
#else
            candidates = new[] { "clang++", "g++" };
#endif
            foreach (string name in candidates)
            {
                try
                {
                    var psi = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = name, Arguments = "--version",
                        UseShellExecute = false, RedirectStandardOutput = true,
                        RedirectStandardError = true, CreateNoWindow = true
                    };
                    using (var p = System.Diagnostics.Process.Start(psi))
                    {
                        p.WaitForExit(3000);
                        if (p.ExitCode == 0) return name;
                    }
                }
                catch { /* not found, try next */ }
            }
            return null;
        }

#if UNITY_EDITOR_WIN
        static string FindMsvcCompiler()
        {
            // Use vswhere to locate MSVC cl.exe
            string vswhere = Path.Combine(
                System.Environment.GetFolderPath(System.Environment.SpecialFolder.ProgramFilesX86),
                "Microsoft Visual Studio/Installer/vswhere.exe");
            if (!System.IO.File.Exists(vswhere)) return null;

            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = vswhere,
                    Arguments = "-latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 " +
                                "-property installationPath",
                    UseShellExecute = false, RedirectStandardOutput = true,
                    CreateNoWindow = true
                };
                using (var p = System.Diagnostics.Process.Start(psi))
                {
                    string vsPath = p.StandardOutput.ReadToEnd().Trim();
                    p.WaitForExit(5000);
                    if (string.IsNullOrEmpty(vsPath)) return null;

                    string vcToolsDir = Path.Combine(vsPath, "VC/Tools/MSVC");
                    if (!Directory.Exists(vcToolsDir)) return null;

                    var versions = Directory.GetDirectories(vcToolsDir);
                    if (versions.Length == 0) return null;

                    System.Array.Sort(versions);
                    string latest = versions[versions.Length - 1];
                    string cl = Path.Combine(latest, "bin/Hostx64/x64/cl.exe");
                    return System.IO.File.Exists(cl) ? cl : null;
                }
            }
            catch { return null; }
        }
#endif

        // Async build state
        static System.Collections.Generic.List<(string label, System.Diagnostics.Process proc, string outAssetPath,
            System.Text.StringBuilder stdout, System.Text.StringBuilder stderr)> _activeBuilds;
        static int _totalBuilds;

        static void StartAsyncBuilds(
            System.Collections.Generic.List<(string label, string exe, string args, string outAssetPath)> builds)
        {
            _activeBuilds = new();
            _totalBuilds = builds.Count;

            foreach (var (label, exe, args, outAssetPath) in builds)
            {
                try
                {
                    string fileName = exe;
                    string arguments = args;

                    // .cmd/.bat files on Windows cannot be started directly with
                    // UseShellExecute=false; route through cmd.exe instead.
                    if (exe.EndsWith(".cmd", System.StringComparison.OrdinalIgnoreCase) ||
                        exe.EndsWith(".bat", System.StringComparison.OrdinalIgnoreCase))
                    {
                        fileName = "cmd.exe";
                        arguments = $"/c \"\"{exe}\" {args}\"";
                    }

                    var psi = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = fileName,
                        Arguments = arguments,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    var proc = System.Diagnostics.Process.Start(psi);
                    var stdoutBuf = new System.Text.StringBuilder();
                    var stderrBuf = new System.Text.StringBuilder();
                    proc.OutputDataReceived += (_, e) => { if (e.Data != null) stdoutBuf.AppendLine(e.Data); };
                    proc.ErrorDataReceived += (_, e) => { if (e.Data != null) stderrBuf.AppendLine(e.Data); };
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    _activeBuilds.Add((label, proc, outAssetPath, stdoutBuf, stderrBuf));
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[RoomScan Setup] Failed to start {label}: {e.Message}");
                }
            }

            if (_activeBuilds.Count == 0)
            {
                Debug.LogError("[RoomScan Setup] No builds started");
                return;
            }

            EditorApplication.update += PollXAtlasBuilds;
            EditorUtility.DisplayProgressBar("Building xatlas", "Compiling native plugins...", 0f);
        }

        static void PollXAtlasBuilds()
        {
            if (_activeBuilds == null) return;

            int done = 0;
            foreach (var (label, proc, _, _, _) in _activeBuilds)
                if (proc.HasExited) done++;

            float progress = (float)done / _totalBuilds;
            string building = done < _totalBuilds
                ? $"Compiling... ({done}/{_totalBuilds} done)"
                : "Finishing...";
            EditorUtility.DisplayProgressBar("Building xatlas", building, progress);

            if (done < _totalBuilds) return;

            // All done
            EditorApplication.update -= PollXAtlasBuilds;
            EditorUtility.ClearProgressBar();

            bool allOk = true;
            var results = new System.Text.StringBuilder();

            foreach (var (label, proc, outAssetPath, _, stderrBuf) in _activeBuilds)
            {
                string stderr = stderrBuf.ToString();
                bool ok = proc.ExitCode == 0;
                allOk &= ok;
                results.AppendLine($"  {label}: {(ok ? "OK" : $"FAILED (exit {proc.ExitCode})")}");

                if (!ok)
                    Debug.LogError($"[RoomScan Setup] {label} build failed (exit {proc.ExitCode}):\n{stderr}");
                else if (!string.IsNullOrWhiteSpace(stderr))
                    Debug.LogWarning($"[RoomScan Setup] {label} warnings:\n{stderr}");
                else
                    Debug.Log($"[RoomScan Setup] {label} build succeeded");

                proc.Dispose();
            }

            AssetDatabase.Refresh();

            // Configure plugin importers after AssetDatabase sees the new files
            EditorApplication.delayCall += () =>
            {
                foreach (var (label, _, outAssetPath, _, _) in _activeBuilds)
                    ConfigurePluginImporter(outAssetPath);
                _activeBuilds = null;
            };

            if (allOk)
                Debug.Log($"[RoomScan Setup] xatlas build complete:\n{results}");
        }

        static void ConfigurePluginImporter(string assetPath)
        {
            var importer = AssetImporter.GetAtPath(assetPath) as PluginImporter;
            if (importer == null)
            {
                Debug.LogWarning($"[RoomScan Setup] PluginImporter not found for {assetPath}");
                return;
            }

            bool isAndroid = assetPath.Contains("/Android/");
            bool isWindows = assetPath.Contains("/Windows/");
            bool isLinux = assetPath.Contains("/Linux/");

            importer.SetCompatibleWithAnyPlatform(false);
            importer.SetCompatibleWithEditor(!isAndroid);
            importer.SetCompatibleWithPlatform(BuildTarget.Android, isAndroid);

            string platformLabel;
            if (isAndroid)
            {
                importer.SetPlatformData(BuildTarget.Android, "CPU", "ARM64");
                platformLabel = "Android ARM64";
            }
            else if (isWindows)
            {
                importer.SetCompatibleWithPlatform(BuildTarget.StandaloneWindows64, true);
                importer.SetEditorData("CPU", "AnyCPU");
                importer.SetEditorData("OS", "Windows");
                platformLabel = "Windows Editor";
            }
            else if (isLinux)
            {
                importer.SetCompatibleWithPlatform(BuildTarget.StandaloneLinux64, true);
                importer.SetEditorData("CPU", "AnyCPU");
                importer.SetEditorData("OS", "Linux");
                platformLabel = "Linux Editor";
            }
            else
            {
                importer.SetCompatibleWithPlatform(BuildTarget.StandaloneOSX, true);
                importer.SetEditorData("CPU", "AnyCPU");
                importer.SetEditorData("OS", "OSX");
                platformLabel = "macOS Editor";
            }

            importer.SaveAndReimport();
            Debug.Log($"[RoomScan Setup] Configured plugin importer: {assetPath} ({platformLabel})");
        }

        // -- Master Button ------------------------------------------------

        void DrawMasterButton()
        {
            var style = new GUIStyle(GUI.skin.button)
            {
                fontStyle = FontStyle.Bold,
                fontSize = 14,
                fixedHeight = 36
            };

            using (new EditorGUI.DisabledScope(_gameReadyFixInProgress))
            {
                if (GUILayout.Button("\u2261  Setup Everything", style))
                    SetupEverything();
            }
        }

        async void SetupEverything()
        {
            if (_gameReadyFixInProgress) return;

            _gameReadyFixInProgress = true;
            try
            {
                if (TrySwitchToAndroidBuildTarget("Setup Everything")) return;

                EditorUtility.DisplayProgressBar("Setup Everything",
                    "Fixing VR prerequisites (Outstanding + Recommended)\u2026", 0.05f);
                VRProjectBootstrap.Audit();

                // URP must exist before anything else so shaders resolve.
                EditorUtility.DisplayProgressBar("Setup Everything",
                    "Ensuring URP pipeline + Quest-friendly defaults\u2026", 0.10f);
                EnsureURPSetup();

                await VRProjectBootstrap.FixAllAsync(CheckSeverity.Recommended);

                // Camera Rig + Passthrough via Meta XR Building Blocks
                // — does the right thing whether or not a rig is already
                // present. Done before AR session so AROcclusionManager
                // can attach to the rig camera.
                EditorUtility.DisplayProgressBar("Setup Everything",
                    "Installing Meta XR Building Blocks (Camera Rig + Passthrough)\u2026", 0.30f);
                await EnsureRequiredBuildingBlocksAsync();
                Refresh();

                EditorUtility.DisplayProgressBar("Setup Everything",
                    "Setting up AR session + occlusion\u2026", 0.35f);
                if (_arSession == null) FixARSession();
                if (_arOcclusion == null) FixAROcclusion();

                // See the matching comment in FixGameReadyModules — run
                // unconditionally so this restores anything OVRProjectSetup
                // stripped during the Recommended VR fix pass above.
                EditorUtility.DisplayProgressBar("Setup Everything",
                    "Updating AndroidManifest + Player Settings\u2026", 0.50f);
                EnsureQuestVRManifest();
                if (!_cleartextAllowed) FixCleartextTraffic();
                if (!_insecureHttpAllowed)
                {
                    PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
                    Debug.Log("[RoomScan Setup] Set Player Settings > insecureHttpOption to AlwaysAllowed");
                }

                EditorUtility.DisplayProgressBar("Setup Everything",
                    "Adding all components + wiring shaders\u2026", 0.75f);
                FixComponents();
                FixShaderWiring();

                RefreshNativePlugins();
                bool xatlasMissing = !_xatlasAndroid || !_xatlasEditor;
                if (xatlasMissing)
                {
                    EditorUtility.DisplayProgressBar("Setup Everything",
                        "Starting xatlas plugin build (background)\u2026", 0.95f);
                    BuildXAtlasPlugin();
                }

                Debug.Log("[RoomScan Setup] Scene setup complete." +
                    (xatlasMissing ? " (xatlas build running in background)" : ""));
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] Setup Everything failed: {ex}");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                _gameReadyFixInProgress = false;
                MarkDirty();
                Refresh();
                Repaint();
            }
        }

        // =================================================================
        //  GUI HELPERS
        // =================================================================

        void BeginSection(string title)
        {
            GUILayout.Space(6);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none,
                GUILayout.ExpandWidth(true), GUILayout.Height(22));
            EditorGUI.DrawRect(rect, COL_SECT);
            var labelRect = new Rect(rect.x + 8, rect.y + 2, rect.width - 16, rect.height);
            var prev = GUI.color;
            GUI.color = Color.white;
            GUI.Label(labelRect, title, EditorStyles.boldLabel);
            GUI.color = prev;
        }

        static void EndSection() => GUILayout.Space(2);

        void StatusRow(string label, bool ok)
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(12);

            string icon = ok ? "\u2713" : "\u2717";
            Color col = ok ? COL_OK : COL_MISS;
            string detail = ok ? "OK" : "Missing";

            var prev = GUI.color;
            GUI.color = col;
            GUILayout.Label(icon, EditorStyles.boldLabel, GUILayout.Width(18));
            GUI.color = prev;

            GUILayout.Label(label, GUILayout.ExpandWidth(true));

            prev = GUI.color;
            GUI.color = col;
            GUILayout.Label(detail, EditorStyles.miniLabel, GUILayout.Width(60));
            GUI.color = prev;

            EditorGUILayout.EndHorizontal();
        }

        void StatusRowOptional(string label, bool attached)
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(12);

            string icon = attached ? "\u2713" : "\u2022";
            Color col = attached ? COL_OK : COL_INFO;
            string detail = attached ? "OK" : "Not Added";

            var prev = GUI.color;
            GUI.color = col;
            GUILayout.Label(icon, EditorStyles.boldLabel, GUILayout.Width(18));
            GUI.color = prev;

            GUILayout.Label(label, GUILayout.ExpandWidth(true));

            prev = GUI.color;
            GUI.color = col;
            GUILayout.Label(detail, EditorStyles.miniLabel, GUILayout.Width(60));
            GUI.color = prev;

            EditorGUILayout.EndHorizontal();
        }

        // =================================================================
        //  UTILITY
        // =================================================================

        static T FindAny<T>() where T : Object =>
            Object.FindAnyObjectByType<T>(FindObjectsInactive.Include);

        static Component FindComponentByTypeName(string typeName)
        {
            foreach (var root in SceneRoots())
            {
                var found = root.GetComponentsInChildren<Component>(true)
                    .FirstOrDefault(c => c != null && c.GetType().Name == typeName);
                if (found != null) return found;
            }
            return null;
        }

        static bool AreFieldsAssigned(Object target, params string[] fieldNames)
        {
            var so = new SerializedObject(target);
            foreach (string name in fieldNames)
            {
                var prop = so.FindProperty(name);
                if (prop == null || prop.objectReferenceValue == null)
                    return false;
            }
            return true;
        }

        static GameObject FindByName(string exact)
        {
            foreach (var root in SceneRoots())
            {
                var t = DeepFind(root.transform,
                    tr => tr.name.Equals(exact, System.StringComparison.Ordinal));
                if (t != null) return t.gameObject;
            }
            return null;
        }

        static Transform DeepFind(Transform root, System.Func<Transform, bool> pred)
        {
            if (pred(root)) return root;
            for (int i = 0; i < root.childCount; i++)
            {
                var hit = DeepFind(root.GetChild(i), pred);
                if (hit != null) return hit;
            }
            return null;
        }

        static GameObject[] SceneRoots() =>
            UnityEngine.SceneManagement.SceneManager.GetActiveScene().GetRootGameObjects();

        static void MarkDirty() =>
            EditorSceneManager.MarkSceneDirty(
                UnityEngine.SceneManagement.SceneManager.GetActiveScene());

        /// <summary>
        /// Sets m_RenderMode to 1 (WorldSpace). The public renderMode property
        /// and PanelRenderMode enum are internal in Unity 6000.3.
        /// </summary>
        static void SetPanelRenderModeWorldSpace(PanelSettings panel)
        {
            const int WorldSpace = 1;
            var so = new SerializedObject(panel);
            var prop = so.FindProperty("m_RenderMode");
            if (prop != null && prop.intValue != WorldSpace)
            {
                prop.intValue = WorldSpace;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(panel);
            }
        }

        internal static void EnsureDebugMenuAssets()
        {
            var ctrl = FindAny<DebugMenuController>();
            if (ctrl == null) return;

            var uiDoc = ctrl.GetComponent<UIDocument>();
            if (uiDoc == null) return;

            Undo.RecordObject(uiDoc, "Assign DebugMenu UIDocument assets");

            if (uiDoc.visualTreeAsset == null)
            {
                var uxml = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(
                    "Packages/com.genesis.roomscan/Runtime/UI/DebugMenu.uxml");
                if (uxml != null) uiDoc.visualTreeAsset = uxml;
            }

            if (uiDoc.panelSettings == null)
            {
                var panel = FindOrCreatePanelSettings();
                if (panel != null) uiDoc.panelSettings = panel;
            }

            // Ensure PanelSettings is configured for world-space VR rendering.
            // renderMode / PanelRenderMode are internal in 6000.3; use SerializedObject.
            if (uiDoc.panelSettings != null)
                SetPanelRenderModeWorldSpace(uiDoc.panelSettings);

            // World-space UIDocument properties:
            // - Dynamic size mode: panel auto-sizes to the content layout (480×640 from USS)
            // - Pivot = Center: transform position = center of the visible panel
            // - PivotReferenceSize = Layout: pivot calculated from root element layout, not bounding box
            uiDoc.worldSpaceSizeMode = WorldSpaceSizeMode.Dynamic;
            uiDoc.pivot = Pivot.Center;
            uiDoc.pivotReferenceSize = PivotReferenceSize.Layout;

            // 480px / 100 PPU = 4.8 local units. Scale 0.08 → 0.384m wide.
            const float worldScale = 0.08f;
            if (Mathf.Abs(ctrl.transform.localScale.x - worldScale) > 0.01f)
            {
                Undo.RecordObject(ctrl.transform, "Scale DebugMenu for VR");
                ctrl.transform.localScale = Vector3.one * worldScale;
            }

            EditorUtility.SetDirty(uiDoc);
        }

        static PanelSettings FindOrCreatePanelSettings()
        {
            const string assetName = "DebugMenuPanelSettings";

            string[] guids = AssetDatabase.FindAssets($"t:PanelSettings {assetName}");
            if (guids.Length > 0)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                var existing = AssetDatabase.LoadAssetAtPath<PanelSettings>(path);
                if (existing != null) return existing;
            }

            const string dir = "Assets/Settings";
            if (!AssetDatabase.IsValidFolder(dir))
                AssetDatabase.CreateFolder("Assets", "Settings");

            const string assetPath = dir + "/" + assetName + ".asset";
            var panel = ScriptableObject.CreateInstance<PanelSettings>();
            AssetDatabase.CreateAsset(panel, assetPath);
            SetPanelRenderModeWorldSpace(panel);
            AssetDatabase.SaveAssets();
            Debug.Log($"[RoomScanWizard] Created PanelSettings (WorldSpace) at {assetPath}");
            return panel;
        }

        static string GetLanIp()
        {
            try
            {
                foreach (var ni in System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces())
                {
                    if (ni.OperationalStatus != System.Net.NetworkInformation.OperationalStatus.Up)
                        continue;
                    if (ni.NetworkInterfaceType == System.Net.NetworkInformation.NetworkInterfaceType.Loopback)
                        continue;

                    var props = ni.GetIPProperties();
                    foreach (var addr in props.UnicastAddresses)
                    {
                        if (addr.Address.AddressFamily != System.Net.Sockets.AddressFamily.InterNetwork)
                            continue;
                        string ip = addr.Address.ToString();
                        if (ip.StartsWith("127.")) continue;
                        return ip;
                    }
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[RoomScanWizard] Failed to detect LAN IP: {e.Message}");
            }
            return null;
        }
    }
}
