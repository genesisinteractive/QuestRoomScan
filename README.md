# QuestRoomScan

Real-time 3D room reconstruction on Meta Quest 3. Produces a textured mesh from depth + RGB camera data using GPU TSDF volume integration and Surface Nets mesh extraction, with server-based Gaussian Splat training and on-device rendering via [Unity Gaussian Splatting](https://github.com/genesisinteractive/UnityGaussianSplatting).

| Scanning (Triplanar) | Vertex Colors |
|:---:|:---:|
| ![Scanning](docs/tsdf2.gif) | ![Vertex Colors](docs/vertex2.gif) |

| Texture Refinement | Gaussian Splat |
|:---:|:---:|
| ![Refined](docs/refined3.gif) | ![Splat](docs/splat.gif) |

**[Full demo video](https://www.youtube.com/watch?v=lEn3GkH7Yao)** — recorded before multi-view blending, GPU sharpening, and atlas enhancement were added; current output quality is noticeably better.

**If you're integrating this package into a game or app**, start with [Installation](#installation), [Quick Start](#quick-start), and [Game Integration Guide](#game-integration-guide). For the full scan pipeline and debug tooling, see [Usage Flow](#usage-flow) and [VR Debug Menu](#vr-debug-menu).

## Table of Contents

- [Used In](#used-in)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Flow](#usage-flow)
  - [Scanning](#scanning)
  - [Freeze / Unfreeze](#freeze--unfreeze)
  - [Training Gaussian Splats](#training-gaussian-splats)
  - [Texture Refinement](#texture-refinement)
  - [Atlas Enhancement (HQ Refine)](#atlas-enhancement-hq-refine)
  - [Mesh Enhancement](#mesh-enhancement)
  - [Saving and Loading](#saving-and-loading)
  - [Architecture](#architecture)
- [Gaussian Splat Pipeline](#gaussian-splat-pipeline)
- [VR Debug Menu](#vr-debug-menu)
- [Memory Budget (Quest 3)](#memory-budget-quest-3)
- [Comparison with Hyperscape](#comparison-with-hyperscape)
- [**Game Integration Guide**](#game-integration-guide)
  - [Placing Content So It Comes Back in the Right Spot](#placing-content-so-it-comes-back-in-the-right-spot)
- [Credits & Prior Art](#credits--prior-art)
- [License](#license)

## Used In

**[CoasterMania](https://www.meta.com/en-gb/experiences/coastermania/7856648691073700/)** — Meta Quest. The **Room Scan Update**, released 1 August 2026, added *Environmental Scanning*: you shrink to track height and ride your rollercoaster in first person through your actual room.

This is the case the package was built for. Quest's built-in room mesh gives you geometry but no surface colour, so when CoasterMania first added a first-person ride in 2023 it had to drop a default green-brown terrain texture over the room and the ride happened in a stand-in of your space rather than your space. Reconstructing a *textured* mesh is what closes that gap.

- Developer post — [true first-person mode, r/VRGaming](https://www.reddit.com/r/VRGaming/comments/1vgmyhy/i_recently_added_a_true_first_person_mode_to_my/)
- Developer comment — [r/OculusQuest](https://www.reddit.com/r/OculusQuest/comments/1vcqem3/comment/p2ro430/?context=3)
- Gameplay — [CoasterMania | I Built A Roller Coaster In My Basement... And Rode It!](https://www.youtube.com/watch?v=31pwtov8hwA), by A Wolf in VR

*Shipped something with this package? Open an issue or a PR and it goes on the list.*

## Features

- **GPU TSDF Integration** — Depth frames fused into a signed distance field via compute shaders
- **GPU Surface Nets Meshing** — Fully GPU-driven mesh extraction via compute shaders with zero CPU readback, rendered via a single `Graphics.RenderPrimitivesIndirect` draw call
- **Two-Layer Real-Time Texturing** — Triplanar world-space cache (~8mm/texel persistent surface color from passthrough RGB) with vertex color fallback (~5cm). Triplanar can be disabled via inspector toggle to save ~192MB GPU memory when not needed (e.g., if only post-scan refined textures matter). Keyframes captured as motion-gated JPEGs to disk for texture refinement and Gaussian Splat training.
- **Package-Based Persistence** — Multi-scan persistence system where each scan is a self-contained package (`pkg_YYYYMMDD_HHMMSS/`) with its own TSDF, triplanar textures, keyframes, splat, and refined textures. Scan browser in the debug menu lists all saved packages. Artifacts (splat, refined, HQ) auto-save to the active package on creation.
- **OVRSpatialAnchor Relocation** — `RoomAnchorManager` creates a persisted `OVRSpatialAnchor` per scan package for reliable cross-session relocation. The package also stores the Scene API UUID of the MRUK room that contained that anchor (`sceneRoomUuid` in `anchor.json` / the manifest) so hosts can test the headset against **that** room, not any captured space. Per-artifact creation matrices in `anchor.json` track when each artifact was created relative to the spatial anchor, enabling accurate relocation even for artifacts created across different sessions. Falls back to MRUK floor anchor if spatial anchor localization fails.
- **Temporal Stabilization** — Adaptive per-vertex temporal blending on GPU prevents mesh jitter while allowing fast convergence. Optional live-mesh **birth fade** and **hold-and-morph** are presentation-only (forward shader). Texture refinement / PLY call `ExtractForAuthoring` and project onto extractor `pos`, never the in-flight lerp.
- **Exclusion Zones** — Capsule rejection around the operator (torso 0.35 m + hand/forearm) so the body is not reconstructed; freeze skips those voxels. Optional Meta depth hand-removal. Up to 64 capsules.
- **Analytic Scan Progress** — every TSDF voxel is observed-free, observed-solid or unknown; unknown that connects to the outside of the volume is *exterior*. Free–solid faces are the surface, free–exterior faces are **leaks** — exactly where passthrough shows through the mesh. `Closure = surface / (surface + leak)`; `ScanCoverage.LeakAreaM2` is the absolute area still open, `HoleCount` / `LargestHole` list the patches, small ones are capped automatically, and the mesh tints them red — all from one GPU label volume, so they agree. Refinement is the share of surface voxels whose TSDF weight says they have stopped moving. Time-sliced GPU work, one 128-byte readback per second; no camera pose, no rays, no scene model.
- **Shell Coverage** — With `RoomUnderstanding`, the captured room hull is sampled into ≤ 16k cells and marched against the TSDF once a second: `ScanCoverage.ShellCoverage` is how much of the shell has scanned matter in front of it (doorways, doors, windows excluded), `LargestGap` / `CopyShellGaps` say where the unscanned patches are, and small wall / floor / ceiling gaps whose neighbours lie on one plane are auto-filled with a soft plane stamp that real depth can still override. Guidance and acceleration only — it never feeds progress.
- **Gaussian Splat Training & Rendering** — Keyframe capture + point cloud export → PC server training → trained PLY download → on-device UGS rendering
- **VR Debug Menu** — Two-panel world-space UI Toolkit HUD with left navigation (Scan, Saved Scans, Refine, Gaussian Splat, Tools) and right detail views. Includes scan browser with load/delete per package (with delete confirmation), "Load Refined Only" for fast game-mode loading, context-sensitive artifact deletion, and dynamic button disabled states. Scene Objects toggle with live count. Navigation tabs for Refine and Gaussian Splat are automatically disabled when their respective modules are not attached.
- **Texture Refinement** — Post-scan UV atlas from captured keyframes. GPU compute shader bakes one thread per atlas texel. Score is head-on × working-distance (peak 0.8–2 m), not `N·V / distance`: standing frontal views beat close-ups when both exist; close-ups still fill holes. Multi-view blend is hero-only and top-K, with per-keyframe pose registration, occlusion-aware depth testing, GPU unsharp-mask sharpening, seam levelling, and Sobel normal maps. `TextureRefinement` is an instance-based MonoBehaviour — unwrap, bake, sharpen and seam settings are inspector fields.
- **Atlas Enhancement (HQ Refine)** — Server-side atlas super-resolution via Real-ESRGAN (2x/4x configurable) + LaMa inpainting. Uploads the on-device refined atlas as PNG, enhances, and downloads the result. Configurable SR scale via inspector.
- **Mesh Enhancement** — Server-side mesh smoothing via bilateral normal filter + optional RANSAC plane detection and vertex snapping. Enhanced mesh saved as a separate artifact preserving the original refined mesh.
- **Render Mode Switching** — Cycle between Wireframe, Vertex, Triplanar, Refined, Occlusion, Splat, and None at runtime via debug menu or controller binding (default: A/X button). Unavailable modes are automatically skipped during cycling (e.g., Triplanar requires `TriplanarCache`, Occlusion/Refined require refinement, Splat requires trained data).
- **Freeze Tint Toggle** — Independent toggle (not tied to render mode) shows/hides a blue tint overlay on frozen voxels in live mesh modes (Vertex, Triplanar, Wireframe). Bindable via `RoomScanInputHandler`.
- **Game Integration APIs** — `RoomScanSession` provides a high-level facade auto-installed by the Game-Ready preset: `RequestCameraPermissionAsync()` / `RequestScenePermissionAsync()` / `RequestAnchorPermissionAsync()` → `WaitUntilRoomReadyAsync()` → `StartScanAsync()` → `FreezeInView()` / `UnfreezeInView()` → `await FinalizeScanAsync()` → `ScanResult` with mesh + atlas. `LoadAsync(packageId)` / `LoadLatestAsync()` for later launches. `UnloadActiveScanAsync()` drops the in-memory mesh and spatial-anchor bind without deleting saved packages (needed before a new scan in the same session). `ListSavedScans()` / `DeleteScanAsync(id)` for games that keep several packages; `ClearAllScansAsync()` for a nuclear wipe. `HasSceneRooms` + `IsHeadsetInsideASceneRoom` + `RequestSpaceSetupAndReloadAsync()` for hosts that want to offer Horizon Space Setup themselves — `RoomAnchorManager` loads the scene with `requestSceneCaptureIfNoDataFound: false` so a missing model does **not** pause into Meta's UI. `HasSceneRooms` only means MRUK loaded *a* room. Native `GetCurrentRoom()` returns the last captured room when you leave, and `IsPositionInRoom` is the floor outline (still true just past a doorway). `IsHeadsetInsideASceneRoom` requires the headset inward of every outer wall plane, including invisible doorway faces — a hallway next to a captured room is outside. A loaded package stores that room's Scene API UUID next to the spatial-anchor UUID (`BoundSceneRoomUuid` / `IsHeadsetInsideBoundSceneRoom`) so a look can hide when the headset is in a different captured room. After spatial-data permission is granted, `ReloadSceneFromDeviceAsync()` re-runs discovery without opening Space Setup (the first load often finished with zero rooms while `USE_SCENE` was still denied). `ConfineScanToContainingRoom` (default off) skips TSDF outside that room when `RoomUnderstanding` is attached. MRUK `SCREEN` planes are always stamped as analytic TSDF slabs (TV glass depth is ignored; RGB still projects). For finer control: `LoadRefinedOnlyAsync()` (loads only refined mesh + atlas, no TSDF, < 1 second), `ReleaseScanResources()`, public `RefinedMesh`/`RefinedAtlas` properties, `RefinedMeshReady` event, and `ScanCoverage`/`ScanProgress` metrics for guided UX. Scene understanding accessible via `SceneObjectRegistry` for MRUK + AI detected objects.
- **Mesh Simplification** — `simplifyBeforeUnwrap` (default on): `meshopt_simplify` reduces the extracted mesh before xatlas, so unwrap and bake run on the game mesh; the dense mesh still builds occlusion. Off: unwrap and bake the dense mesh, then `meshopt_simplifyWithAttributes` with UV-locked borders into `simplified_mesh.bin`. Keyframes are registered to the first-pass atlas (ZNCC shift → rotation), exposure-equalised, and blended with a per-chart preferred view and seam levelling.
- **AI Object Detection** — Optional YOLO-based object detection via Unity Inference Engine (Sentis) running during scanning. GPU Non-Maximum Suppression via compute shader (only ~500 bytes readback vs ~200KB for CPU NMS). Detected objects projected to 3D world space via GPU depth projection with temporal snapshot to handle async inference. Head angular velocity gating skips blurry frames. Detection keyframes saved with JSONL metadata for post-processing.
- **MRUK Scene Understanding** — `RoomUnderstanding` is the MRUK wrapper: occupancy (headset inside a captured room / a specific Scene API UUID), vertical scene planes filtered by host `SceneFaceKind`, classification, and `SceneObjectRegistry` population (walls, floor, ceiling, bed, TV, doors, windows, furniture). During a scan, `SCREEN` anchors become analytic TSDF plane stamps via a dedicated voxel-AABB dispatch (depth on glass is discarded; RGB is kept). `CopyRoomClipPlanes` / `CopyRoomWorldAabb` feed the opt-in single-room TSDF clip. Without this component the scan is unbounded (no clip, no SCREEN stamps) and occupancy APIs on `RoomScanSession` return false / empty — permissions and MRUK load still live on `RoomAnchorManager` / `DepthCapture`. Never use native `GetCurrentRoom()` — it is last/first after you leave; occupancy walks every loaded room's outer wall planes. Hosts still go through `RoomScanSession` (`IsHeadsetInsideASceneRoom`, `CopyHeadsetRoomWallFaces`, `ConfineScanToContainingRoom`, …). Uses `SceneModel.V2FallbackV1` with high-fidelity scene mesh. Event-driven anchor updates.
- **Scene Object Debug Visualization** — Toggle world-space wireframe bounding boxes + billboard labels for all detected objects (MRUK + AI). Rendered via `DebugOverlay.shader` with per-source color coding (cyan = MRUK, yellow = AI). Count shown in debug menu button.
- **Sobel Normal Maps** — GPU Sobel edge detection in `AtlasBakeCompute.compute` generates normal maps from the baked atlas. `RefinedMesh.shader` uses Sobel normals for real-time lighting on the refined mesh, adding depth and surface detail.

## Requirements

- **Unity 6** (6000.x)
- **URP** (Universal Render Pipeline)
- **Meta Quest 3** (depth sensor required)

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| `com.unity.xr.arfoundation` | 6.1+ | Depth frame access |
| `com.unity.render-pipelines.universal` | 17.0+ | URP rendering pipeline |
| `com.meta.xr.mrutilitykit` | 205+ | Passthrough camera RGB access |
| `com.unity.burst` | 1.8+ | Required by Collections/Mathematics |
| `com.unity.collections` | 2.4+ | NativeArray for plane detection |
| `com.unity.mathematics` | 1.3+ | Math types used throughout |
| `org.nesnausk.gaussian-splatting` | [fork](https://github.com/genesisinteractive/UnityGaussianSplatting) | **Optional** — Gaussian splat rendering with runtime PLY loading |
| `com.unity.ai.inference` | 2.x+ | **Optional** — AI object detection (YOLO via Sentis). Assembly `Genesis.RoomScan.AIDetection` auto-activates when present |

Additional project-level dependencies (not in `package.json` — installed via Meta's SDK or XR plugin management):
- `com.unity.xr.meta-openxr` (bridges Meta depth to AR Foundation)
- `com.unity.xr.openxr` (OpenXR runtime)
- `com.meta.xr.sdk.core` (OVRInput, OVRCameraRig)

### Android Permissions

- `com.oculus.permission.USE_SCENE` (depth API / spatial data)
- `horizonos.permission.HEADSET_CAMERA` (passthrough camera RGB access)
- `com.oculus.permission.USE_ANCHOR_API` (spatial anchors)

## Installation

Add to your project's `Packages/manifest.json`, pinned to a release tag:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "https://github.com/genesisinteractive/QuestRoomScan.git#v1.3.1"
  }
}
```

Drop the `#v1.3.1` suffix to track `main`. Releases and their notes are in
[`CHANGELOG.md`](CHANGELOG.md); `main` only moves by squash-merged release PR.

For Gaussian Splat support, also add the optional dependency:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "https://github.com/genesisinteractive/QuestRoomScan.git",
    "org.nesnausk.gaussian-splatting": "https://github.com/genesisinteractive/UnityGaussianSplatting.git?path=package#main"
  }
}
```

For AI object detection (YOLO), add the Sentis inference package:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "https://github.com/genesisinteractive/QuestRoomScan.git",
    "com.unity.ai.inference": "2.3.0"
  }
}
```

Both optional dependencies can be combined. The `Genesis.RoomScan.AIDetection` assembly auto-activates when `com.unity.ai.inference` is installed (via `HAS_AI_INFERENCE` define).

## Quick Start

1. Create a new blank URP scene (or use an existing one).
2. Open the setup wizard: **`RoomScan > Setup Scene`**.
3. Click **`Apply Game-Ready Setup`** at the top of the wizard. One click does the following — there's nothing else you need to configure first:
   - Switches the active build profile to **Meta Quest** (re-click after the domain reload to continue with the rest)
   - Creates and assigns a **URP** pipeline + renderer at `Assets/Settings/` with Quest-friendly defaults (4x MSAA, no HDR, single shadow cascade)
   - Audits and fixes ~20 **VR project prerequisites**: XR Plug-in Management (OpenXR loader on Android), OpenXR feature groups (Meta XR, Touch interaction profile), `OVRProjectConfig` (Quest 3 target devices, scene support, passthrough, hand tracking, anchors), `OVRRuntimeSettings`, scripting backend (IL2CPP / ARM64), graphics (Vulkan-only, single-pass-instanced stereo), and ASTC texture compression
   - Writes `Assets/Plugins/Android/AndroidManifest.xml` with the full Quest VR feature/permission set (`HEADSET_CAMERA`, `USE_SCENE`, `USE_ANCHOR_API`, `BOUNDARYLESS`, etc.) plus cleartext HTTP for the LAN GS server, without clobbering anything already there
   - Installs Meta XR Building Blocks (**OVRCameraRig**, **Passthrough Underlay**, **PassthroughCameraAccess**) and configures `OVRManager` for passthrough + transparent center-eye camera + camera permission on startup
   - Adds AR Session + AROcclusionManager on the new camera rig
   - Adds the lean game-ready scene modules to a `RoomScan` root: `RoomScanner`, `RoomScanPersistence`, `RoomAnchorManager`, `PassthroughCameraProvider`, `TextureRefinement` (with 0.5 post-bake simplification), `RoomUnderstanding`, `RoomScanSession`. Skips `TriplanarCache`, Gaussian Splat, and debug HUD to keep the build lean
   - Wires every shader / compute / material reference and triggers the xatlas native plugin build in the background
4. The first click typically lands you on Meta Quest profile and reloads the domain. Click **`Apply Game-Ready Setup`** again to finish project + scene setup. Repeat until every status row is green.
5. (Optional) Apply the **`Add Debug Tools`** block lower in the wizard if you want the world-space VR debug HUD, controller input handler, camera/depth overlays, and VR input infrastructure for development. Keep it skipped for shipping builds.
6. (Optional) Add modules not in the Game-Ready preset (`TriplanarCache`, `GSplatManager`, `ObjectDetectionModule`) via the inspector's **Add Module** dropdown on the `RoomScanner` component.
7. The wizard also compiles native xatlas plugins (required for texture refinement). On **macOS**, this uses the system `clang++`. On **Windows**, you need Visual Studio with the C++ Desktop workload (for the editor plugin) — the Android plugin uses Unity's bundled NDK and needs no extra install. If the build fails, retry from the **Build xatlas Plugin** button in the wizard's NATIVE PLUGINS section.
8. Build and deploy to Quest 3.
9. The room mesh appears as you look around — surfaces solidify with repeated observations.

> **The wizard is idempotent.** Re-running `Apply Game-Ready Setup` on an already-set-up project only fixes outstanding rows; it never duplicates components or overwrites your edits to assigned fields. If a row is red after running, click again — the most common cause is a domain reload split across two clicks.

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Black screen / no passthrough | Meta XR feature group not enabled | Re-run `Apply Game-Ready Setup` until the `XR Plug-in Management (OpenXR + Meta XR feature group)` row goes green |
| Black screen / no passthrough | Camera background not transparent | Re-run `Apply Game-Ready Setup` until `Passthrough scene config (OVRManager + transparent center camera + HEADSET_CAMERA on startup)` is green |
| App launches but no permission dialog appears | A second `RequestUserPermission` while another is in flight is dropped by Android with no UI | Every request in this package goes through one serialised queue (`StartScanAsync` asks for scene → camera → anchors; `RoomScanSession.Request*PermissionAsync` joins the same queue). Keep `OVRManager.request*PermissionOnStartup` **off** — it requests outside that queue; `Apply Game-Ready Setup` turns it off. Do not call `Permission.RequestUserPermission` yourself for these three. |
| Controller ray visible but no UI | Debug menu not opened | Press **left thumbstick click** to toggle the debug menu (debug-tools build only) |
| Scanning stays at "Discovering" | Depth frames not arriving | Verify the wizard's `VR PROJECT BOOTSTRAP` panel is fully green; check that `com.oculus.permission.USE_SCENE` is in `AndroidManifest.xml` |
| Refine shows "--" / does nothing | xatlas native plugin not built | Open the wizard and click **Build xatlas Plugin** in the NATIVE PLUGINS section. On Windows, requires Visual Studio C++ workload or clang++ on PATH |
| xatlas build fails on Windows | macOS-only build (pre-v1.x) | Pull latest main — Windows/Linux support was added |
| Wall of AR / Meta XR errors when hitting Play in Editor without Quest Link | `ARSession` / `AROcclusionManager` / `PassthroughCameraAccess` react to the missing XR loader | Expected and not silenceable from outside Meta's package — every workaround introduced its own NRE chain. Build to a Quest 3, or attach via Quest Link, to actually test scanning. Editor play mode is fine for non-XR work; the errors are noise |

## Usage Flow

### Scanning

Call `await RoomScanner.Instance.StartScanningAsync()` to begin (or use the debug menu). As you look around:

1. **Depth integration**: Each depth frame is fused into the TSDF volume with color from the passthrough camera
2. **Mesh extraction**: GPU Surface Nets extracts a mesh from the volume every few frames (after a minimum number of integrations)
3. **Texturing**: Camera RGB is baked into triplanar world-space textures for persistent surface color (with vertex color fallback)
4. **Keyframe capture**: Motion-gated JPEG snapshots + camera poses are saved into the active scan package on disk — these are used later for Gaussian Splat training
5. **Point cloud export**: GPU mesh vertices exported as `points3d.ply` on demand (before GS training or via debug menu)

**Tips for a good scan**: Move slowly around the room. Look at surfaces from multiple angles — repeated observations from different viewpoints improve mesh quality. Make sure to cover walls, floor, ceiling, and furniture from several directions before training.

### Freeze / Unfreeze

When a region of the mesh looks good and you don't want further integration to degrade it:

- **Freeze In View** (Y/B button): Locks the voxels inside a 15° spotlight cone from your eye along your gaze (`freezeConeHalfAngle`); turn your head to paint more. Frozen voxels are skipped during integration — their geometry and color are preserved exactly as-is. Hosts can instead pass a custom cone: `FreezeInView(origin, direction, halfAngleDegrees, maxMetres)` (length 0 is unbounded).
- **Unfreeze In View** (X/A button): Restores frozen voxels in the same cone to normal integration. Same optional custom-cone overload.

This lets you selectively protect good surfaces while continuing to refine other areas.

### Training Gaussian Splats

Once the room is well-scanned:

1. Open the debug menu (left thumbstick click)
2. Verify the **Server URL** points to your PC running [RoomScan-GaussianSplatServer](https://github.com/genesisinteractive/RoomScan-GaussianSplatServer). If you used the setup wizard and your PC is on the same LAN, the IP is auto-detected and should already be correct. For a cloud/remote server, edit the URL in the debug menu or set it in the Inspector before building.
3. Press **Start GS Training** — this triggers the full pipeline automatically:
   - Exports the current mesh as a point cloud (`points3d.ply`)
   - ZIPs all keyframes, poses, and point cloud from the active package
   - Uploads the ZIP to the server
   - The debug menu shows live training status: state, progress bar, iteration count, elapsed time, backend
   - When training completes, the trained PLY is downloaded back to the Quest
4. Press **Render Mode** to cycle to Splat view — the downloaded PLY is loaded into `GaussianSplatRenderer` and rendered on-device
5. Cycle through render modes (Wireframe → Vertex → Triplanar → Refined → Occlusion → Splat → None) to compare views — modes whose data is not present are skipped automatically

Scanning continues during training — you can keep refining the mesh while waiting.

### Texture Refinement

After scanning, you can produce a sharper UV-mapped texture atlas from the captured keyframes:

1. Open the debug menu
2. Press **Refine Textures** — this runs the full on-device pipeline:
   - **GPU readback**: Reads the current mesh from the GPU Surface Nets buffers
   - **UV unwrapping**: xatlas (native C++ via P/Invoke) generates a UV atlas with seam-aware parameterization, with tunable chart/pack options for speed vs quality
   - **GPU atlas baking**: A compute shader (`AtlasBakeCompute.compute`) processes each keyframe — two-pass multi-view blending selects and blends the top-scoring views per texel with occlusion-aware depth testing (~5-10s for 300 keyframes)
   - **Seam levelling**: the two atlas sides of every UV seam edge are paired by 3D position, pinned to their mean and the correction diffused into each chart on the GPU; per-keyframe exposure gains and a smooth view-admission ramp keep view switches inside a chart from printing a line
   - **GPU sharpening**: Unsharp mask restores crispness lost during multi-view blending (configurable strength and radius)
   - **Sobel normal map**: GPU Sobel edge detection generates a normal map from the atlas for real-time fake lighting
   - **Dilation**: Fills gaps at UV island edges
3. Press **Render Mode** to cycle to **Refined** — the UV-mapped mesh with baked atlas texture and normal-mapped lighting

Refined textures persist automatically with the active package and are restored on load.

### Atlas Enhancement (HQ Refine)

For further quality improvement, the on-device atlas can be enhanced via a server-side pipeline:

1. Press **HQ Refine (Server)** in the debug menu (auto-triggers on-device refinement if not done)
2. The refined atlas is encoded as PNG, uploaded to the server
3. Server applies **Real-ESRGAN** super-resolution (2x/4x, configurable via `hqRefineScale`) + **LaMa** inpainting to fill gaps
4. Enhanced atlas is downloaded and applied

The SR scale is configurable in the inspector. Requires a server running at the configured URL.

> **Note:** An earlier differentiable-rendering-based HQ path is non-functional and not exposed. The Real-ESRGAN + LaMa pipeline described above is the working path.

### Mesh Enhancement

Server-side mesh geometry enhancement:

1. Press **Enhance Mesh (Server)** in the Refine view
2. The refined mesh is serialized and uploaded to the server
3. Server applies bilateral normal filter (smoothing) + optional RANSAC plane detection and vertex snapping
4. Enhanced mesh is downloaded and displayed, saved as a **separate artifact** (`enhanced_mesh.bin`) — the original refined mesh is preserved
5. **Delete Enhanced Mesh** in the debug menu restores the original refined mesh

### Saving and Loading

QuestRoomScan uses a **package-based persistence system**. Each scan is saved as a self-contained package under `RoomScans/`:

```
RoomScans/
  manifest.json
  pkg_20260228_143022/
    scan.bin              # TSDF + color volumes (v1 binary)
    anchor.json           # Spatial-anchor UUID + scene-room UUID + per-artifact matrices
    triplanar/            # Color + depth textures (saved when triplanar is enabled)
    keyframes/            # Motion-gated keyframes (images/ + frames.jsonl)
    splat.ply             # Auto-saved when GS training completes
    refined_mesh.bin      # Auto-saved when on-device refinement completes
    refined_atlas.raw     # Auto-saved with refined mesh
    simplified_mesh.bin   # Optional: post-bake simplified copy when simplifyBeforeUnwrap is off
    enhanced_mesh.bin     # Auto-saved when server mesh enhancement completes
    hq_atlas.png          # Auto-saved when server atlas enhancement completes
```

- **Save Scan**: Promotes the temporary scan package to a permanent package. Persists the TSDF + color volumes, triplanar textures (when enabled), and creates a persisted `OVRSpatialAnchor` for cross-session relocation. Keyframes are already in-place from scanning. Sets this package as the active target for subsequent artifact auto-saves. Saving is disabled while scanning is active.
- **Load Scan**: Browse saved packages in the **Saved Scans** view. Loading a package localizes the spatial anchor, computes per-artifact relocation matrices, and restores all data including splat, refined textures, enhanced mesh, and HQ atlas.
- **Auto-save artifacts**: When a splat download completes, on-device refinement finishes, atlas/mesh enhancement finishes, the artifact is automatically saved to the active package — no manual "Save Scan" needed.
- **Delete artifact**: Context-sensitive deletion in the Scan view — deletes the artifact matching the current render mode (splat, refined, enhanced mesh, or HQ) from the active package.
- **Delete package**: Full package deletion from the Saved Scans view, including erasing the spatial anchor from persistent storage.
- **Clear All Data**: Stops scanning, clears volumes/mesh/keyframes from memory, cleans up any temporary package, clears the active package reference.

**Data flow**: When scanning starts, a temporary package (`_tmp`) is created. All keyframes and artifacts write directly into it. On save, `_tmp` is atomically promoted to a permanent package — no file copying needed. If the app crashes, `_tmp` is cleaned up on next startup.

### Architecture

The package follows a **modular architecture**. Core components are always required; optional modules can be added via the RoomScanner inspector's "Add Module" dropdown.

**Core (always required):**

```
RoomScanner (orchestrator, events, scan lifecycle)
  ├── DepthCapture (AROcclusionManager → depth → normals → dilation, tracking→world)
  ├── VolumeIntegrator (TSDF + color integration, exclusion zones, prune, freeze)
  ├── MeshExtractor → GPUSurfaceNets → GPUMeshRenderer (fully GPU-driven mesh)
  ├── RoomScanPersistence (package-based multi-scan persistence)
  └── RoomAnchorManager (OVRSpatialAnchor relocation)
```

**Optional modules (add via inspector):**

```
  ├── PassthroughCameraProvider (RGB frames from headset cameras)
  ├── TriplanarCache (bake camera RGB → 3 world-space textures + depth maps)
  ├── TextureRefinement (GPU readback → xatlas UV unwrap → multi-view atlas bake + Sobel normals)
  │     └── requires KeyframeCollector (auto-added)
  ├── RoomUnderstanding (MRUK occupancy, wall faces, classification → SceneObjectRegistry)
  ├── SceneObjectVisualizer (world-space wireframe boxes + billboard labels for detected objects)
  ├── RoomScanSession (high-level facade for game integration — see Game Integration Guide)
  ├── [separate assembly] GSplatManager + GSplatServerClient (Gaussian Splat training & rendering)
  │     └── requires KeyframeCollector (auto-added)
  └── [separate assembly] ObjectDetectionModule + YoloDetectionModel (AI detection via Sentis)
        └── requires PassthroughCameraProvider
```

All optional modules implement `IRoomScanModule` and are discovered automatically at startup. The `GaussianSplatting` package dependency lives in the separate `Genesis.RoomScan.GSplat` assembly, and AI detection lives in `Genesis.RoomScan.AIDetection` — consumers who don't need either can omit them entirely.

See [ALGORITHM.md](ALGORITHM.md) for the full technical reference.

## Gaussian Splat Pipeline

QuestRoomScan captures keyframes and a dense point cloud during scanning, uploads them to a PC training server, and renders the trained Gaussian splats on-device. See [Usage Flow > Training Gaussian Splats](#training-gaussian-splats) for the step-by-step user guide.

### On-Device Capture (automatic during scanning)

- **KeyframeCollector**: Motion-gated JPEG frames + camera poses saved directly into the active scan package (`keyframes/images/*.jpg`, `keyframes/frames.jsonl`). Captures are triggered by camera movement — you get more keyframes by looking at the room from different angles.
- **PointCloudExporter**: GPU mesh vertices exported as binary PLY (`points3d.ply`) via `AsyncGPUReadback`. Exported on demand — automatically before GS training upload, or manually via the debug menu's Tools view.

### Server Training (via [RoomScan-GaussianSplatServer](https://github.com/genesisinteractive/RoomScan-GaussianSplatServer))

The companion PC server handles the full training pipeline:

```bash
python main.py --port 8420  # API server
npm run dev                  # Dashboard at http://localhost:5173
```

When you press **Start GS Training** in the debug menu, the following happens automatically:

1. **Export**: Final point cloud exported from GPU mesh
2. **ZIP & Upload**: Quest packages the active scan's keyframe directory (`frames.jsonl`, `points3d.ply`, `images/*.jpg`) into a ZIP and POSTs to `{serverUrl}/upload?iterations={N}`
3. **Convert**: Server converts Unity poses + intrinsics to COLMAP binary format, computes scene normalization
4. **Train**: Gaussian Splat training via msplat (Metal), gsplat (CUDA), or 3DGS — the debug menu shows live progress
5. **Denormalize**: Output PLY is transformed back to world coordinates (reverses nerfstudio-style scene normalization)
6. **Download**: Quest GETs `{serverUrl}/download` → trained PLY bytes stored in memory
7. **View**: Press **Render Mode** to cycle to Splat — `GSplatManager` loads PLY via `GaussianSplatPlyLoader.LoadFromPlyBytes()` and renders on-device

### On-Device Rendering (UGS)

Trained splats are rendered using a [fork of Unity Gaussian Splatting](https://github.com/genesisinteractive/UnityGaussianSplatting) with runtime PLY loading and Quest 3 optimizations:

- **`GaussianSplatPlyLoader`**: Parses binary PLY → converts to UGS internal format → creates GPU buffers directly (no Editor asset pipeline needed)
- **Coordinate conversion**: COLMAP (right-handed Y-down) → Unity (left-handed Y-up)
- **Quest 3 stereo**: Per-eye VP matrices for correct VR covariance projection, shared compute between eyes
- **Performance**: Reduced-resolution rendering (0.5x), optimized compute shaders, partial radix sort, contribution-based culling
- **Render mode switching**: Cycled via debug menu or controller binding without releasing GPU resources. Available modes: Wireframe, Vertex, Triplanar, Refined, Occlusion, Splat, None — unavailable modes are skipped

### Supported Training Backends

| Backend | Platform | Install |
|---------|----------|---------|
| [msplat](https://github.com/nicknish/msplat) | Apple Silicon (Metal) | `pip install "msplat[cli]"` |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | NVIDIA GPU (CUDA) | `pip install gsplat` |
| [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | NVIDIA GPU (CUDA) | Clone repo, pass `--gs-repo` |

## VR Debug Menu

Two-panel world-space UI Toolkit panel activated via **left thumbstick click**. Point the controller ray at buttons and press the **index trigger** to click. The panel lazy-follows your gaze at 0.75m.

### Layout

```
+------------------+---------------------------------------------+
| QUESTROOMSCAN    |  [Right panel — swaps based on nav]          |
|  DEBUG           |                                             |
|                  |                                             |
| [*] Scan         |  (Scan View / Saved Scans / Refine /       |
| [ ] Saved Scans  |   Gaussian Splat / Tools)                    |
| [ ] Refine       |                                             |
| [ ] Gaussian Splat|                                             |
| [ ] Tools        |                                             |
|                  |                                             |
| 72 FPS           |                                             |
+------------------+---------------------------------------------+
```

> **Module-gated tabs:** The Refine and Gaussian Splat navigation tabs are automatically disabled (dimmed and non-clickable) when `TextureRefinement` or `GSplatManager` modules are not attached to the RoomScanner.

### Views

**Scan** (default) — Live status rows (Scanning, Mode, Integrations, Keyframes, Render, Package) plus coverage metrics (Progress, Phase, Color Coverage, Frozen, Mesh Stats) and action buttons:
- **Start/Stop Scanning**: Toggle depth integration
- **Render Mode**: Cycle through Wireframe → Vertex → Triplanar → Refined → Occlusion → Splat → None (unavailable modes skipped)
- **Freeze Tint**: Toggle blue tint overlay on frozen voxels (works in Vertex, Triplanar, and Wireframe modes)
- **Objects**: Toggle world-space debug visualization of detected objects (MRUK + AI). Button text shows live count (e.g., "Objects: ON (10M + 2AI)")
- **Save Scan**: Create a new package with current scan data
- **Delete Artifact**: Context-sensitive — deletes Splat/Refined/HQ atlas from active package based on current render mode

**Saved Scans** — Scrollable list of saved packages sorted newest-first. Each entry shows display name, date, artifact badges (KF, Tri, Splat, Refined, HQ, Enh), and Load/Ref (load refined only)/Delete buttons. Delete requires two-click confirmation (turns red with "Sure?" text, resets after 3 seconds). Badge count shown on the nav button.

**Refine** — On-device and server refinement status, mesh stats (original refined and simplified vertex/tri counts), and action buttons. Tab is disabled when `TextureRefinement` module is not attached.
- **Refine Textures**: On-device GPU atlas bake from keyframes (multi-view blend + sharpen)
- **HQ Refine (Server)**: Upload atlas for server-side super-resolution + inpainting
- **Enhance Mesh (Server)**: Upload mesh for server-side bilateral smooth + plane snap

**Gaussian Splat** — GS training with server URL field, live progress, and Start/Cancel buttons. Tab is disabled when `GSplatManager` module is not attached.

**Tools** — Export Point Cloud, Clear All Data.

### Button Disabled States

Buttons are dynamically enabled/disabled based on app context:
- **Save Scan**: Disabled if no volume data
- **Start GS Training**: Disabled if already training or no scan loaded
- **Refine Textures**: Disabled if already refining or no mesh/keyframes
- **HQ Refine**: Disabled if no server URL configured
- **Export Point Cloud**: Disabled if no volume data
- **Delete Artifact**: Only visible in Splat/Refined/HQRefined modes, requires active package. If enhanced mesh exists in Refined mode, deletes enhanced mesh first.
- **Enhance Mesh**: Disabled if no refined mesh or already enhancing

### Default Controller Bindings

| Button | Action |
|--------|--------|
| Left Thumbstick Click | Toggle Debug Menu |
| One (Y/B) | Freeze In View |
| Two (X/A) | Unfreeze In View |
| Three (A/X) | Cycle Render Mode |
| Four (B/Y) | Start Server Training (disabled by default) |

Additional bindable actions (not mapped by default): `ToggleFreezeTint`, `ToggleScanning`, `SaveScan`, `LoadScan`, `ClearAllData`, `ExportPointCloud`.

All bindings are configurable via `RoomScanInputHandler` — add, remove, or remap any `ScanAction` to any `OVRInput.Button`.

## Memory Budget (Quest 3)

Default values — all configurable per-component in the Inspector.

| Component | Default | Memory |
|-----------|---------|--------|
| TSDF volume (RG8_SNorm) | 256 x 256 x 256 | ~32 MB |
| Color volume (RGBA8) | 256 x 256 x 256 | ~64 MB |
| GPU Surface Nets (coord map, vertices, indices, smoothing, temporal 3D texture) | 256³ derived | ~83 MB |
| Triplanar color textures (3x RGBA8) | 3 x 4096 x 4096 | ~192 MB |
| Triplanar depth textures (3x R8) | 3 x 4096 x 4096 | ~48 MB |
| **Total GPU** | | **~419 MB** |

**Disabling triplanar** (`TriplanarCache.enableTriplanar = false` in inspector) drops the total to **~179 MB** by skipping all six texture allocations. The mesh falls back to vertex colors (~5cm resolution), which are still adequate for real-time scanning visualization. This is a good option when:

- You only care about the post-scan refined texture (which is significantly sharper than triplanar)
- You're running additional GPU-heavy workloads alongside scanning
- You want to maximize headroom on Quest 3's shared GPU memory

Keyframes are written as JPEGs to disk (not held in GPU memory). To further reduce GPU memory, lower `VolumeIntegrator.voxelCount` in the Inspector.

## Comparison with Hyperscape

[Meta Horizon Hyperscape](https://www.meta.com/help/quest/1088536553019177/) is Meta's first-party room scanning app for Quest 3. It produces stunning photorealistic Gaussian Splat captures — significantly higher visual quality than what QuestRoomScan currently achieves. If your goal is purely the best-looking scan, Hyperscape is the better choice today.

QuestRoomScan exists for a different reason: it's **open source, fully on-device, and gives you complete control over the pipeline**.

| | Hyperscape | QuestRoomScan |
|-|------------|---------------|
| **Processing** | Cloud (1-8 hours after capture) | Real-time textured mesh on-device, GS training on local PC |
| **Output quality** | Photorealistic Gaussian Splats | Textured mesh (real-time) + on-device GS rendering via UGS |
| **Data access** | No raw file export | Full export: PLY point cloud, JPEG keyframes, camera poses |
| **Extensibility** | Closed, no API | MIT open source, every parameter exposed |
| **GS training** | Handled by Meta's cloud | Your hardware, your choice of backend (msplat/gsplat/3DGS) |
| **Offline use** | Requires upload + cloud processing | Works entirely offline (except GS training on PC) |
| **Integration** | Standalone app | Unity package — embed scanning in your own app |

QuestRoomScan is best suited for developers who need to integrate room scanning into their own applications, want full control over the reconstruction pipeline, or need to work with the raw scan data directly.

## Game Integration Guide

This section covers how to embed QuestRoomScan into a game that needs a one-time room scan followed by lightweight rendering.

### Quick Start with RoomScanSession

The simplest integration uses `RoomScanSession` — a high-level facade that wraps the full scan → refine → save → release flow into a few awaitable calls. The `Apply Game-Ready Setup` wizard preset adds it to the `RoomScan` root automatically.

```csharp
var session = RoomScanSession.Instance;

// 0. (Optional) Ask for permissions up front so you control when the OS
//    sheets appear and can show your own "asking" state. StartScanAsync
//    requests anything still missing (scene, then camera, then anchors)
//    through the same serialised queue, so this is UX, not correctness:
//    a scan started without this still gets its dialogs. Returns true
//    immediately if already granted or off-Android.
if (!session.HasScenePermission && !await session.RequestScenePermissionAsync())
    { /* spatial data is required to scan; tell the user, offer Settings */ return; }
if (!session.HasCameraPermission && !await session.RequestCameraPermissionAsync())
    { /* optional: scanning proceeds depth-only without textures */ }

// 1. (Optional) Single-scan games: wipe any previous saved scan so the
//    on-device scan store doesn't grow ~100 MB per finalize.
if (session.HasSavedScan)
    await session.ClearAllScansAsync();

// 2. Begin scanning. The room mesh builds in real-time as the user looks around.
//    Awaitable: StartScanAsync stages the ~600 MB GPU bring-up across ~4 yielded
//    frames (~56 ms total) before enabling the passthrough camera and depth
//    sensor. Below the human-perception threshold for "press registered".
await session.StartScanAsync();
session.ProgressUpdated += p => progressBar.value = p.OverallProgress;

// 3. As the user sweeps the room, paint visible chunks as "done":
//    FreezeInView locks voxels in a spotlight cone so they stop receiving
//    updates. The no-arg path is the headset gaze. Hosts with their own
//    emitter pass origin, axis, half-angle, and length (0 = unbounded).
session.FreezeInView();    // typically bound to a controller button
session.UnfreezeInView();  // for "I painted too aggressively, redo"

// 4. When the user is done, commit:
ScanResult result = await session.FinalizeScanAsync();
// result.Mesh      — simplified UV-mapped mesh, ready for MeshFilter
// result.Atlas     — baked texture atlas, ready for material.mainTexture
// result.PackageId — save this if you want to reload a specific scan later
// GPU resources already released (~400-500 MB freed)

// Subsequent launches: skip scanning entirely
if (session.HasSavedScan)
{
    ScanResult result = await session.LoadLatestAsync();
    // or: await session.LoadAsync(savedPackageId);
    // Mesh + atlas ready in < 1 second, relocated via the saved spatial anchor
}
```

`FinalizeScanAsync()` handles everything: stop scanning → texture refinement → save to disk → release GPU resources. The scan is also persisted as a self-contained package under `Application.persistentDataPath/RoomScans/pkg_<timestamp>/` with its own `OVRSpatialAnchor` for cross-session relocation.

> **Who asks for permissions.** Every runtime-permission request in this package goes through one serialised queue: `StartScanAsync` asks for `USE_SCENE` (required — a denial aborts the start), then `HEADSET_CAMERA` and `USE_ANCHOR_API` (a denial degrades), *before* any GPU bring-up, so a scan never starts half-permitted and no two dialogs race. `RoomScanSession.Request*PermissionAsync` joins that same queue, which is how a host front-loads the sheets at boot and shows its own "asking" state; once granted, the requests inside `StartScanAsync` are free. Android drops a second `RequestUserPermission` while one is up — with no UI — so do not add your own `Permission.RequestUserPermission` for these three, and keep `OVRManager.request*PermissionOnStartup` off (the Game-Ready preset does).

> **Why `ClearAllScansAsync` and not save-then-purge.** `ClearAllScansAsync` deliberately wipes only saved packages (`pkg_*/` and `manifest.json`), never the active `_tmp/` working directory — that one is owned by `StartScan` / `FinalizeScanAsync`'s lifecycle. Safe to call at any point: if nothing is saved it returns immediately. The trade-off is that a finalize crash after a `ClearAllScansAsync` loses the previous scan; if your game wants the old scan to outlive a finalize failure, save first and purge after.

### Lifecycle: Scan Phase → Game Phase

For developers who want finer control, the underlying APIs are:

```
1. Scan Phase:    await StartScanningAsync() → user looks around → StopScanning()
2. Refine:        StartTextureRefinement() → wait for RefinedMeshReady event
3. Transition:    ReleaseScanResources() → frees ~400-500 MB GPU memory
4. Game Phase:    Render with standard MeshRenderer (1 draw call, baked texture)
```

On subsequent launches, skip scanning entirely:

```
1. LoadRefinedOnlyAsync(pkgId) → loads refined mesh + atlas in < 1 second
2. Game Phase immediately
```

### Placing Content So It Comes Back in the Right Spot

Unity's world origin is wherever the headset booted, so world coordinates mean a
different physical place on every run. The spatial anchor is the only transform
that re-localizes to the same spot, so it is the only frame worth saving against.

**Parenting under the anchor is not by itself enough.** `SetParent(anchor,
worldPositionStays: true)` keeps the child's world pose and stores the difference
as a local offset. That correctly tracks drift correction — the usual reason to
parent — but leaves local space equal to world space plus a constant, so local
coordinates still mean a different place next run. Within one session this is
invisible, which is what makes it a trap: content is perfect on the run that
authored it and wrong on every run after.

There are two correct recipes, and which one you need depends on whether a
transform can move your data.

#### Recipe 1 — content you can parent (use this by default)

Put a `RoomSpaceRoot` on a scene root GameObject. It binds itself to the spatial
anchor and holds its own local transform at identity, so **its local space is the
anchor's space**. Save local coordinates; reload them; done. No matrices.

```csharp
// Spawn into room space (or RoomSpaceRoot.Adopt for an existing GameObject).
var tree = RoomSpaceRoot.Spawn(treePrefab, worldPos, worldRot);

// Persist: local coordinates are room coordinates.
save.position = tree.transform.localPosition;
save.rotation = tree.transform.localRotation;

// Restore on a later run, after the anchor has bound.
if (await RoomSpaceRoot.WaitForBindAsync())
{
    var restored = Instantiate(treePrefab);
    RoomSpaceRoot.Adopt(restored);
    restored.transform.localPosition = save.position;
    restored.transform.localRotation = save.rotation;
}
```

When the anchor binds or changes, direct children keep their world poses, so
nothing visibly jumps; their local coordinates are re-expressed in the new frame.
Before any anchor binds, the root sits at the world origin and `IsBound` is
false — content placed then is correct for the session but its coordinates are
not yet room coordinates, so await `WaitForBindAsync()` before persisting.

If you generate geometry rather than place prefabs, build vertices through
`RoomSpaceRoot.WorldToRoom` and attach the result with `AdoptAtRoomOrigin`, which
avoids applying the room transform twice.

#### Recipe 2 — data with world coordinates baked in

A transform cannot move vertices already written into an array. For those, store
`RoomAnchorManager.Instance.SpatialAnchorMatrix` at bake time and apply the delta
on load — which is exactly how the refined scan mesh itself survives a restart:

```csharp
// At bake time, beside the data:
save.anchorAtBake = RoomAnchorManager.Instance.SpatialAnchorMatrix;

// On load:
var reloc = RoomAnchorManager.ComputeRelocationMatrix(
    RoomAnchorManager.Instance.SpatialAnchorMatrix, save.anchorAtBake);
for (int i = 0; i < positions.Length; i++)
    positions[i] = reloc.MultiplyPoint3x4(positions[i]);
```

Record the anchor matrix in the same operation that bakes the data. Assigning it
afterwards is the common bug: any code that draws the content in between will use
whatever the field was initialised to, and identity is a legal frame that gets
replayed as a full anchor-pose offset. For the same reason, use a sentinel that
is *not* identity for "never recorded" — the zero matrix works, since it fails
`Matrix4x4.ValidTRS()` and is also what a missing JSON field deserializes to.

> **Breaking change.** Earlier versions expected consumers to parent their own
> root under `SpatialAnchorTransform` with `worldPositionStays: true`. That
> produces content which is correctly placed in the session that authored it and
> misplaced afterwards. Migrating to `RoomSpaceRoot` changes what previously
> saved local coordinates mean, so bump your save format and regenerate rather
> than loading old data into the new frame.

### Recommended Configuration

For game integration where you want to minimize GPU overhead during scanning. These are also the field defaults (Game-Ready Apply restamps them onto older scenes):

| Setting | Value | Reason |
|---------|-------|--------|
| RoomScanner.meshExtractionHz | **8** | Live Surface Nets dump; 30 Hz was fill-rate expensive |
| KeyframeCollector move / rotate / interval | **0.15 m / 10° / 0.25 s**, angular velocity **120°/s**; band gap **0.5 m**, standing yaw **8°**; hand-heavy frames kept | Atlas bake needs density and both close-up and standing views |
| TextureRefinement view score | Head-on × 0.8–2 m working distance; close-ups fill holes | Stops 20 cm grazes from beating standing head-on looks |
| TextureRefinement.postBakeSimplificationRatio | **0.5** | Simplified before unwrap and bake; 1.0 disables |
| RoomScanner.ConfineScanToContainingRoom | host opt-in | Single-room mesh; default **false**. Needs `RoomUnderstanding` |
| TriplanarCache | **Disabled** | Saves ~240 MB GPU; vertex colors are sufficient for scan-phase visualization |
| GaussianSplatRenderer | **Not attached** | Remove unless splat rendering is needed |

### API Reference

#### Accessing the Refined Mesh

```csharp
var scanner = RoomScanner.Instance;

// Option 1: Subscribe to the event
scanner.RefinedMeshReady += (mesh, atlas) =>
{
    // mesh: Unity Mesh with UV coordinates
    // atlas: Texture2D with baked texture atlas
    myMeshFilter.mesh = mesh;
    myRenderer.material.mainTexture = atlas;
};

// Option 2: Read properties directly (null until refinement completes)
Mesh mesh = scanner.RefinedMesh;
Texture2D atlas = scanner.RefinedAtlas;
Texture2D hqAtlas = scanner.HQAtlas; // null if no server enhancement
```

#### Lightweight Loading (Game Sessions)

```csharp
// Save the package ID after scanning
string pkgId = scanner.Persistence.ActivePackageId;

// On next launch — loads only refined mesh + atlas, no TSDF/Surface Nets
bool ok = await RoomScanner.Instance.LoadRefinedOnlyAsync(pkgId);
// Mesh is now visible with standard MeshRenderer, render mode auto-set to Refined
```

#### Releasing GPU Resources

```csharp
// After scanning + refinement, before entering gameplay
scanner.ReleaseScanResources();
// Frees ~400-500 MB (TSDF volumes, Surface Nets buffers, depth textures)
// Refined MeshRenderer stays alive for game-phase rendering
// Vertex/Wireframe/Triplanar modes become unavailable (IsModeAvailable returns false)

// To scan again later (re-allocates everything):
await scanner.StartScanningAsync();
```

#### Monitoring Scan Progress

```csharp
// Analytic coverage: read off the live mesh and the TSDF, no scene model needed
ScanCoverage cov = scanner.CurrentCoverage;
Debug.Log($"Closed: {cov.Closure:P0} (holes {cov.HoleCount}, leak {cov.LeakAreaM2:F2} m²), " +
          $"Refined: {cov.Refinement:P0}, " +
          $"Largest hole at {cov.LargestHole.Center} ({cov.LargestHole.AreaM2:F2} m²)");

// High-level progress = Closure × (1 − 0.4 × (1 − Refinement)); gate on cov.LeakAreaM2
ScanProgress prog = scanner.CurrentProgress;
progressBar.value = prog.OverallProgress; // 0.0 – 1.0
statusText.text = prog.Phase.ToString();  // Discovering → Refining → Stabilized → Complete
```

#### Mesh Simplification (before the bake)

Set `TextureRefinement.postBakeSimplificationRatio` in the Inspector (default **0.5** = 50% triangle reduction; 1.0 disables). Simplification runs **before** the UV unwrap, so the atlas is baked onto the mesh that is displayed; the dense mesh is still used for occlusion during the bake. No separate `simplified_mesh.bin` is produced (older packages that have one still load).

#### Keyframe Registration and Chart-Consistent Blend

`refineKeyframePoses` (default on) aligns each keyframe to the first-pass atlas with a low-resolution ZNCC image shift and folds it into the keyframe rotation before blending. `chartBestViewBoost` (default 3) lets the view that scores best over a whole UV chart carry that chart, moving view switches to chart borders. See ALGORITHM.md §15.1a–b.

### `RoomScanSession` API Surface

Everything a game needs lives on one component. `[RequireComponent(typeof(RoomScanner))]` pulls in the scanner; `RoomScanPersistence` is auto-added by the wizard.

| Member | Type | Purpose |
|---|---|---|
| `Instance` | `static RoomScanSession` | Singleton; set in `Awake`, cleared in `OnDestroy` |
| `IsScanning` | `bool` | Live scan state (mirrors `RoomScanner.IsScanning`) |
| `HasSavedScan` | `bool` | True if at least one `pkg_*/` exists on disk |
| `HasCameraPermission` | `bool` | Horizon OS `HEADSET_CAMERA` granted (always true off-Android) |
| `HasScenePermission` | `bool` | Horizon OS `USE_SCENE` (spatial data) granted |
| `HasAnchorPermission` | `bool` | Horizon OS `USE_ANCHOR_API` granted |
| `IsRoomLoaded` | `bool` | MRUK `LoadSceneFromDevice` finished (including zero rooms). All discovery anchors are already on the rooms. |
| `HasSceneRooms` | `bool` | At least one MRUK room after discovery (not "headset is in that room") |
| `IsHeadsetInsideASceneRoom` | `bool` | Headset is inward of every outer wall of **any** loaded room (doorway faces included). Boot / Space Setup: any set-up room is enough. Floor-outline `IsPositionInRoom` is not enough. Always true in the editor |
| `ConfineScanToContainingRoom` | `bool` | When true, TSDF stays in the MRUK room that contained the headset at scan start (outer walls / floor / ceiling expanded 50 cm outward, then hard-confined). Default **false**. Set before `StartScanAsync`. No-op without `RoomUnderstanding` |
| `StampScreenPlanes` | `bool` | When true, `SCREEN` (TV) plane stamps are a dedicated voxel-AABB dispatch after Integrate. Default **true**. Set false before `StartScanAsync` to skip |
| `SetBodyExclusionAnchors(head, left, right)` | `void` | Optional. Pin head + wrist transforms for torso/hand/forearm capsules and mark them host-owned (scanner will not overwrite). For non-OVR rigs. Default is `OVRCameraRig` each integrate. Null wrists skip that side |
| `BoundSceneRoomUuid` | `Guid` | Scene API UUID of the MRUK room the active package was scanned in (stored with the spatial-anchor UUID). Empty when no package is loaded. Rebound from the localized anchor pose if missing or stale |
| `IsHeadsetInsideBoundSceneRoom` | `bool` | Headset is inside the active package's bound room — not some other captured space. False when no package is loaded. Always true in the editor |
| `CopyHeadsetRoomWallFaces(List, SceneFaceKind)` | `int` | Vertical planes of every loaded room that contains the headset. Host passes `SceneFaceKind` (`Wall`, `Screen`, or both). Empty in the editor and when not inside a captured room. |
| `HeadsetSceneRoomUuid` | `Guid` | Scene API UUID of the room that contains the headset, or empty. Not the active scan package (`BoundSceneRoomUuid`) |
| `TryRebindBoundSceneRoomIfHeadsetMatches()` | `bool` | After `LoadAsync`: true when headset and the localized spatial anchor share a captured room; persists that room's current Scene API UUID (Space Setup redo in the same physical room). False in a hallway or a different set-up room |
| `ProgressUpdated` | `event Action<ScanProgress>` | Per-frame progress while scanning. `OverallProgress = Coverage.Closure × (1 − 0.4 × (1 − Coverage.Refinement))` — closure is surface ÷ (surface + leak) from the boundary of observed free space; no camera or scene model involved. `Coverage.LeakAreaM2` is the absolute area still open (the natural gate), `HoleCount` / `LargestHole` locate the patches. With `RoomUnderstanding`, `Coverage.ShellCoverage` / `LargestGap` add hull-based guidance and drive auto-fill, but never the progress number. Hosts gate finalize themselves — the package does not |
| `CopyShellGaps(List<ShellGap>)` | `int` | Largest uncovered shell patches (≤ 8, largest first): centre, normal, cell count, area, surface kind. 0 when shell coverage is unavailable |
| `AutoFillShellGaps` | `bool` | Soft-stamp the captured plane over small wall / floor / ceiling gaps while scanning. Default **true** |
| `RequestCameraPermissionAsync()` | `Task<bool>` | Awaits the system permission dialog; resolves true if already granted |
| `RequestScenePermissionAsync()` | `Task<bool>` | Awaits spatial-data permission |
| `RequestAnchorPermissionAsync()` | `Task<bool>` | Awaits spatial-anchor permission |
| `WaitUntilRoomReadyAsync()` | `Task` | Completes when `LoadSceneFromDevice` has finished. Every scene anchor from that discovery is already present. |
| `RoomReady` | `event Action` | Same moment as `WaitUntilRoomReadyAsync` |
| `SceneAnchorsChanged` | `event Action` | Room or scene anchor created/updated after the load (`RoomUpdatedEvent` / `AnchorCreatedEvent`) |
| `ReloadSceneFromDeviceAsync()` | `Task<bool>` | Re-run discovery with auto-capture **off** (no Space Setup). True if rooms exist. Use after spatial-data permission is granted — the first load often finished empty while `USE_SCENE` was still denied. |
| `RequestSpaceSetupAndReloadAsync()` | `Task<bool>` | Horizon Space Setup, then reload with auto-capture **off**. True only if rooms exist afterwards (cancel is not success-with-rooms) |
| `StartScanAsync()` | `Task` | Begin a new scan session (unloads a loaded package on a non-resume start, creates `_tmp/` package + spatial anchor; completes at the first integrated frame) |
| `FreezeInView()` | `void` | Paint voxels inside the head cone (`FreezeConeHalfAngle`, 15°) as done; integration continues globally |
| `FreezeInView(origin, direction, halfAngleDegrees, maxMetres)` | `void` | Same paint, host-supplied cone. `maxMetres` 0 is unbounded |
| `UnfreezeInView()` | `void` | Inverse of `FreezeInView` for re-capture of bad regions |
| `UnfreezeInView(origin, direction, halfAngleDegrees, maxMetres)` | `void` | Inverse of the custom-cone freeze |
| `FinalizeScanAsync()` | `Task<ScanResult>` | Stop scanning → refine → save; returns mesh + atlas + package id + `AnchorFrameMesh`. Releases the live TSDF when `PresentRefinedWhenReady` is true |
| `PresentRefinedWhenReady` | `bool` | True (default): finalize presents the refined mesh and releases the live TSDF. False: bake into memory and keep the live vertex mesh until the host presents |
| `SetRefinedBackfaceCull(cullBack)` | `void` | Two-sided in the room, `Cull Back` outside (`RefinedMeshBackface.shader`). Quest ignores ShaderLab `Cull [_Cull]` |
| `LoadAsync(packageId)` | `Task<ScanResult>` | Load refined mesh + atlas from a specific package (< 1 s) |
| `LoadLatestAsync()` | `Task<ScanResult>` | Load the newest saved package |
| `UnloadActiveScanAsync()` | `Task` | Drop in-memory mesh + spatial-anchor bind; does **not** delete `pkg_*` |
| `ListSavedScans()` | `IReadOnlyList<SavedScanInfo>` | Every saved package (id, display name, timestamp), newest first |
| `DeleteScanAsync(packageId)` | `Task` | Erase one `pkg_*/` plus its spatial anchor |
| `ClearAllScansAsync()` | `Task` | Erase every saved `pkg_*/`, the manifest, and per-package spatial anchors (leaves `_tmp/` alone) |
| `ReleaseScanResources()` | `void` | Free ~400-500 MB of GPU memory (auto-called by `FinalizeScanAsync`) |

`ScanResult` is `{ Mesh, Texture2D Atlas, string PackageId, Mesh AnchorFrameMesh }`. `AnchorFrameMesh` is the same mesh in the spatial-anchor frame, identical across loads — use it for content persisted relative to the room.

### Minimal Integration Checklist

1. Add QuestRoomScan to your `Packages/manifest.json` (see [Installation](#installation)).
2. Open **`RoomScan > Setup Scene`** and click **`Apply Game-Ready Setup`**. Re-click after the build-target / domain reload to finish.
3. Build to Quest 3 (or attach Quest Link).
4. Game code: `await RoomScanSession.Instance.StartScanAsync();` — it requests scene / camera / anchor permissions itself if they are missing. Call `Request*PermissionAsync` earlier only if you want the OS sheets at a moment of your choosing.
5. Bind a controller button to `FreezeInView` and another to `UnfreezeInView` (the QRS DebugMenu uses Y/B + X/A by default).
6. When the user commits: `var result = await RoomScanSession.Instance.FinalizeScanAsync();` → use `result.Mesh` + `result.Atlas` to render with a standard `MeshRenderer`.
7. On subsequent launches: `if (session.HasSavedScan) await session.LoadLatestAsync();` — skip scanning entirely.

## Credits & Prior Art

The TSDF volume integration and Surface Nets meshing approach draws inspiration from [anaglyphs/lasertag](https://github.com/anaglyphs/lasertag) by Julian Triveri & Hazel Roeder (MIT), which demonstrated real-time room reconstruction on Quest 3 inside a mixed reality game.

The texture refinement pipeline uses two open-source native C++ libraries:

- **[xatlas](https://github.com/jpcy/xatlas)** by Jonathan Young (MIT) — automatic UV atlas generation with seam-aware chart parameterization and efficient packing. Used for UV unwrapping the GPU Surface Nets mesh prior to atlas baking.
- **[meshoptimizer](https://github.com/zeux/meshoptimizer)** v1.0 by Arseny Kapoulkine (MIT) — mesh optimization toolkit. `meshopt_simplify` reduces geometry before unwrap (default); `meshopt_simplifyWithAttributes` with `LockBorder` is the post-bake path. Set `TextureRefinement.postBakeSimplificationRatio` < 1.0 to enable.

Both libraries are compiled into a single native shared library (`libxatlas.so` / `libxatlas.bundle`) and invoked via P/Invoke from C#.

QuestRoomScan builds on that foundation with significant extensions:

| | lasertag | QuestRoomScan |
|-|----------|---------------|
| **Mesh extraction** | CPU marching cubes from GPU volume | Fully GPU-driven Surface Nets via compute shaders — zero CPU readback, single indirect draw call |
| **Texturing** | Geometry only — no camera RGB texturing | Real-time triplanar cache (~8mm/texel) + vertex colors, post-scan refined atlas (keyframe multi-view bake + SR enhancement) |
| **Persistence** | None — mesh lost on restart | Multi-scan package persistence with OVRSpatialAnchor cross-session relocation |
| **Mesh quality** | Basic TSDF blending | Quality² modulation, confidence-gated Surface Nets, warmup clearing, pruning, body exclusion zones, GPU temporal stabilization, RANSAC plane detection & snapping |
| **Gaussian Splatting** | — | Full pipeline: on-device capture → PC server training → on-device UGS rendering with render mode switching |
| **VR UI** | — | World-space debug menu with controller ray interaction, live status, and training controls |
| **Packaging** | Embedded in a game | Standalone Unity package with one-click editor setup wizard |

## License

[MIT](LICENSE.md) — see [LICENSE.md](LICENSE.md) for full text and attribution.
