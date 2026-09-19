# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.3.1] - 2026-09-19

The package now lives at
[genesisinteractive/QuestRoomScan](https://github.com/genesisinteractive/QuestRoomScan);
the previous GitHub URL redirects. Profiled texture bakes log each
compositor hitch as it happens.

### Changed

- Install and documentation URLs point at
  `https://github.com/genesisinteractive/QuestRoomScan`. Pin
  `#v1.3.1`.

### Added

- When `profileRefinement` is on, every compositor frame over 25 ms is
  logged with the current pipeline stage, not only in the end summary.

## [1.3.0] - 2026-09-15

Fixes a **texture-quality regression** from 1.0–1.1: close-up keyframes
were beating standing head-on views. Standing, head-on frames now win
when both exist. Close-up photos stay on disk and still fill holes.
Hosts can freeze and unfreeze with a custom spotlight cone instead of
the headset gaze.

### Added

- **Custom freeze cone.** `RoomScanSession.FreezeInView(origin, direction,
  halfAngleDegrees, maxMetres)` (and the matching unfreeze) paint a
  host-supplied spotlight instead of the headset gaze. Length 0 is
  unbounded. The no-arg `FreezeInView()` / debug-menu path is still the
  15° head cone.

### Fixed

- **Texture-quality regression (1.0–1.1).** Atlas score was `N·V / distance`,
  so a 20 cm graze beat a 1.5 m frontal look; 1.1.0 then admitted only
  those winners. Score is now head-on × working distance (peak 0.8–2 m).
  Blend admission is hero-only; pass 2 walks best-first so the per-texel
  cap is top-K. Chart preference requires a cover floor so a corner
  close-up cannot lock an island. Exposure gains match the standing
  pass-1 atlas (`exposureGainLimit` 1.6).

### Changed

- **Keyframe capture.** Same-pose lean-in / step-back and new wall yaw
  at a given distance band are kept. Frames with hands in view are
  saved (bake still masks capsules per pixel). Optional `"z"` in
  `frames.jsonl`.

## [1.2.0] - 2026-09-14

Texture refinement is much faster and holds **72 fps** on Quest 3
through unwrap and both bake passes. Denser keyframes, a pipelined
GPU atlas, and views that agree across seams. Game APIs for an
anchor-frame mesh, when to present the refined result, and which scene
planes to copy.

### Added

- **Anchor-frame mesh.** `ScanResult.AnchorFrameMesh` and
  `RoomScanPersistence.BuildAnchorFrameMesh` rebuild the game mesh in the
  spatial-anchor frame from package constants only
  (`AnchorAtCreate⁻¹ × stored vertices`). The world mesh is relocated with
  the pose sampled when the anchor localized; a root parented under the
  anchor then moves with tracking, so `worldMesh × root.worldToLocal`
  differs by millimetres on every call and every session. Author persistent
  room content in this frame and present it under `RoomSpaceRoot`.
  `RefinedRelocation` and `RefinedAnchorAtCreate` expose the matrices.
- **`PresentRefinedWhenReady`** (default true) on `RoomScanSession` /
  `RoomScanner`. When false, finalize bakes into memory
  (`HasRefinedTexture`, `RefinedMeshReady`) but keeps drawing the live
  vertex mesh until the host presents `ScanRenderMode.Refined` and calls
  `ReleaseScanResources`.
- **`SceneFaceKind` on plane copy.** `CopyHeadsetRoomWallFaces(dest, kind)`
  takes `Wall`, `Screen`, or both. Overlapping rooms are unioned. Copied
  planes are uniform (no per-label size gate). `RoomScanSession.RoomReady`
  is `LoadSceneFromDevice` finished; `SceneAnchorsChanged` is later
  `RoomUpdated` / `AnchorCreated`.
- **Cull Back refined-mesh shader** (`RefinedMeshBackface.shader`) swapped
  in by `RoomScanSession.SetRefinedBackfaceCull`. Quest ignores ShaderLab
  `Cull [_Cull]`; this is a second program. Default refined shader stays
  two-sided. `OcclusionMesh` is now `Cull Front` so an inward room writes
  depth from outside.
- **`KeyframeImageDecoder`.** File read + JPEG decode in one worker hop:
  Android `BitmapFactory` via JNI into `ARGB_8888`, rows flipped; editor
  and other platforms (or a JNI failure) fall back to `LoadImage`.
- **xatlas threading.** `xatlas::SetThreading(maxThreads, workerNice)`
  (`xatlas_set_threading`). `TextureRefinement.xatlasThreads` (3) and
  `xatlasThreadNice` (10) keep the unwrap off most of Quest 3's cores and
  below the engine's threads. The P/Invoke is guarded for older plugin
  binaries; rebuild from the wizard on Windows / Linux.
- **Optional `ICameraFrameTiming`** on camera providers: capture time of
  `CurrentFrame` on a monotonic clock, so a consumer can measure motion
  between frames from the frames' own timestamps.
- **Refinement profile** (`TextureRefinement.profileRefinement`,
  `KeyframeCollector.profileCapture`; both **off**). One
  `[TextureRefine][Profile]` block per bake (stages, per-keyframe
  main/worker time, compositor-frame count/mean/max, hitches, managed and
  native memory) and a `[KeyframeCollector][Profile]` line every 25 saves
  and at scan stop.

### Changed

- **Keyframe collection** defaults to 0.15 m / 10° / 0.25 s with a 120 °/s
  angular gate on the app clock. Hand capsules are clipped at the near plane
  before projection so a forearm behind the camera is not a full-image
  footprint. Capture encode is `EncodeArrayToJPG` on a worker; the main
  thread only copies the readback.
- **Bake is pipelined.** One keyframe per compositor frame: N+1 is read and
  decoded on a worker (prefetch depth 3) while N's GPU runs; the main thread
  uploads into one of two alternating `Texture2D`s and issues
  clear → depth → shade in one submission. Atlas buffers are zeroed on the
  GPU. Mesh readback requests `vertCount × stride` / `idxCount × 4` and
  parses on a worker. Atlas, normal map and mesh apply over three frames;
  tangents are `TextureRefinement.ComputeTangents` on a worker.
- **Texel-parallel bake.** `BuildTexelMap` once per bake; `BakeAtlas` /
  `BlendAccum` run one thread per atlas texel with a per-texel score (own
  point, interpolated vertex normal). Occlusion depth is photo ÷
  `occlusionDepthDivisor` (2). `SampleKf` is bilinear. `MatchShift` is one
  256-thread group per candidate shift with a groupshared reduction.
  `simplifyBeforeUnwrap` (default on): geometry-only `meshopt_simplify`
  before xatlas (target error 3e-3 of extent); the dense mesh still builds
  occlusion. Off: the 1.1 path — unwrap the dense mesh, UV-locked simplify
  after the bake into `simplified_mesh.bin`.
- **Views agree across a chart.** Pass 1 aligns each keyframe to the atlas
  (`refineKeyframePoses`: 1/4-res ZNCC shift → yaw/pitch). Blend admission
  is a ramp from `blendMinFraction` × best to the best score. The
  chart-preferred view is boosted × `chartBestViewBoost` (3) and exempt
  from `maxViewsPerTexel`, but meets the same bar. With registration on,
  `equalizeExposure` scales each photo to the pass-1 atlas
  (`exposureGainLimit` 1.6). `ResolveBlend` keeps pass-1 colour where no
  blend sample reached.
- **Seam levelling** replaces `BlendSeams`. CPU `BuildSeamPairs` (position-
  deduped endpoints) then GPU `SeamDelta` / `SeamDiffuse` ×
  `seamLevelIterations` (40, one a frame) / `SeamApply`. `enableSeamBlending`
  keeps its name; `seamBlendRadius` is gone.

### Removed

- `SceneWallFace.IsScreen`. Filter with `SceneFaceKind` at copy time.

## [1.1.0] - 2026-09-11

Analytic scan progress, body exclusion that holds up, and a texture bake
that ignores the player's hands. `RoomScanSession` only gains members;
`ScanCoverage` loses its legacy stabilisation fields (see Removed).

### Added

- **Analytic scan progress** from the boundary of observed free space. Every
  TSDF voxel is observed-free, observed-solid or unknown; unknown that
  touches a full 8³ block of unknown (a 40 cm cube of nothing — the outside,
  the far side of a hole, the inside of a couch) is *void*, carried through
  the shell blocks by an LDS fine flood. Free–solid faces are the surface,
  free–void faces are **leaks** — exactly where passthrough shows through the
  mesh. Faces within `cutToleranceVoxels` of a clip plane or the volume edge
  are cuts, not leaks. `Closure = surface / (surface + leak)`;
  `ScanProgress.OverallProgress = Closure × (1 − refinementInfluence × (1 −
  Refinement))`, refinement being the fraction of surface voxels at or above
  `confidentWeight`. One full-volume classify per cycle, everything else over
  the ~10 % of blocks that hold both unknown and observed voxels, time-sliced
  over ~20 frames, one 128-byte readback per `analysisIntervalSeconds`. No
  camera pose, no rays, no scene model.
  `ScanCoverage` gains `AnalysisAvailable`, `Closure`, `Refinement`,
  `ConfidentFraction`, `ConfidentSurfaceCount`, `LeakAreaM2` (the absolute
  number a host should gate on), `SurfaceAreaM2`, `HoleCount`, `LargestHole`
  (`MeshHole`: centre, area, faces), `LeakFills`.
- **Leak tint** (`RoomScanSession.ShowHoles`): the labels live in an
  `R8_UNorm` 3-D texture (`gsLabelVolume`) the scan mesh shader samples per
  vertex, so what is red is what is counted, listed and filled.
- **Leak fill** (`fillLeaks`, on): leak patches up to `fillLeakMaxAreaM2`
  (0.25 m²) are capped by turning the two unknown voxels behind each leak
  face solid, continued from the free voxel's own TSDF value at the band
  slope, so a hole in a wall closes on the wall. Frontier-sized patches are
  never capped; real depth is never overwritten.
- **Body exclusion capsules** replace the 0.6 m head cylinder: torso
  (0.35 m, world-up), hands (0.14 m) and short forearms. Tested at both the
  depth sample (a body pixel integrates nothing along its ray — the negative
  band behind a hand otherwise meshes as a hand-shaped shell) and the voxel
  (a wall behind a hand still fills). `FreezeInView` skips capsules; unfreeze
  does not. Optional `eraseBodyBlobs` (off) clears leftover hand voxels
  below `eraseMaxWeight`. `RoomScanner` refreshes head / wrists from
  `OVRCameraRig` each integrate (controller → `HandOnControllerAnchor`, else
  tracked `OVRHand`); `RoomScanSession.SetBodyExclusionAnchors` is for hosts
  with a non-OVR rig and marks the anchors host-owned.
- `DepthCapture.removeHandsFromDepth` requests Meta occlusion hand removal
  (inpainted depth). Needs hand tracking; the runtime turns it off while
  controllers are held, which the capsules cover.
- **Hands stay out of the texture.** `KeyframeCollector` records the hand /
  forearm capsules at capture (`"cap"` in `frames.jsonl`, relocated with the
  pose on load) and skips frames where they cover more than
  `maxHandCoverage` (12 %) of the image; `AtlasBakeCompute` rejects any
  texel whose line of sight from that view passes through a recorded
  capsule (radius × 1.4), in both the best-view and the blend pass.
- **Shell coverage** (needs `RoomUnderstanding`): the captured hull — outer
  walls, floor, ceiling, furniture faces — is sampled into ≤ 16k cells and
  marched against the TSDF on the analysis tick. `ScanCoverage` gains
  `ShellCoverageAvailable`, `ShellCoverage` (openings excluded from the
  denominator), `ShellCellsTotal / Covered / Excluded / Empty`,
  `ShellGapCount`, `LargestGap`, `ShellFillsApplied`;
  `RoomScanSession.CopyShellGaps` lists the largest unscanned patches.
  Furniture faces march through the whole scene box, and a segment the
  sensor has seen straight through is reported empty and leaves the
  denominator — air inside a loose couch or table box is not a hole.
  Guidance and auto-fill only; it never feeds `OverallProgress`.
- **Shell auto-fill** while scanning: small wall / floor / ceiling gaps whose
  covered neighbours lie on one plane are stamped with that plane at a soft
  weight (`autoFillShellGaps`, on); small furniture gaps get a cluster-local
  6-neighbour close (`closeFurnitureHoles`, on). Real depth overrides both.
- **Freeze / unfreeze spotlight**: `FreezeInView` / `UnfreezeInView` paint a
  head-forward cone (`freezeConeHalfAngle`, 15°) instead of the whole depth
  frustum; `RoomScanSession.FreezeConeHalfAngle` lets a host draw the ring.

### Changed

- Multi-view bake: `blendMinFraction` 0.3 → 0.75 and new `maxViewsPerTexel`
  (3), so a long scan no longer averages dozens of misregistered views into
  mush. Keyframe capture thresholds 0.4 m / 20° → 0.5 m / 25°.
- `ScanProgress.Phase` thresholds now read `OverallProgress` (< 0.30
  Discovering, < 0.90 Refining, < 0.95 Stabilized, else Complete).
- The compute package compiles warning-free on Vulkan (single-exit helpers,
  direction tables instead of dynamic vector-component writes).

### Removed

- The frozen-fraction / colour / vertex-plateau progress blend:
  `ScanCoverage.IsStabilized`, `VolumeIntegrator.coverageUpdateInterval` and
  the separate coverage readback. `FrozenFraction` and `ColorCoverage` stay
  as raw fields. Hosts that gated on `IsStabilized` should gate on
  `Coverage.LeakAreaM2` (absolute) or `OverallProgress`.
- `RoomScanner.TryGetCameraIntrinsics` and the body-exclusion diagnostic log.

## [1.0.0] - 2026-09-09

First stable release. `RoomScanSession` is API-stable from here; breaking
changes bump the major version. `v0.1.0` predates almost everything below,
so this entry describes the package rather than a diff.

### Live scan (GPU only, zero CPU readback)

- TSDF volume integration from the Quest depth sensor (256³, RG8 SNorm +
  RGBA8 colour): bilateral depth filter, dilation, normals, adaptive
  weighting, pruning, exclusion zones around tracked heads.
- GPU Surface Nets mesh extraction in compute, drawn with one
  `Graphics.RenderPrimitivesIndirect`; adaptive per-vertex temporal damping,
  HC Laplacian smoothing, plane-snap regularization. Indirect argument
  buffers are zeroed at allocation so the preview draws nothing until the
  first extraction.
- Two-layer texturing: triplanar world-space colour cache (~8 mm/texel) from
  the passthrough RGB camera, vertex colour fallback. Triplanar optional.
- Presentation-only live-mesh birth fade and hold-and-morph between
  extractions (`GPUVertex` 48 bytes with `prevPos`; authoring paths read
  extractor `pos` via `ExtractForAuthoring`).
- `FreezeInView` / `UnfreezeInView`, `ScanCoverage` / `ScanProgress`,
  render modes Wireframe / Vertex / Triplanar / Refined / Occlusion / Splat /
  None, freeze tint toggle.
- Defaults tuned in a shipped title: extract 8 Hz, keyframes 0.4 m / 20° /
  1 s, post-bake simplify 0.5, vertex budget 8 %, warmup 3.

### Scene API priors (optional `RoomUnderstanding`)

- `ConfineScanToContainingRoom` (default off): TSDF clipped to the MRUK room
  containing the headset — outer walls / floor / ceiling expanded 50 cm
  outward, then hard-confined, with a matching AABB cull; triplanar bake
  honours the same clip.
- `StampScreenPlanes` (default on): MRUK `SCREEN` planes written as analytic
  TSDF slabs via a voxel-AABB dispatch so TV glass is not depth noise.
- Without the module the scan is unbounded and occupancy APIs return
  false / empty; permissions and MRUK load do not depend on it.

### Refinement and persistence

- Texture refinement: xatlas UV unwrap (native plugin; builds on macOS /
  Windows / Linux hosts), multi-view atlas bake from motion-gated keyframes,
  Sobel normal maps, UV-preserving post-bake simplification (meshoptimizer).
- Package-based persistence: `pkg_YYYYMMDD_HHMMSS/` with TSDF, triplanar,
  keyframes, refined mesh + atlas, splat, and a manifest; `_tmp/` staging;
  `LoadRefinedOnlyAsync` loads mesh + atlas in under a second.
- `OVRSpatialAnchor` relocation per package with per-artifact creation
  matrices; each package stores the Scene API UUID of the room it was
  scanned in, rebound from the anchor pose when missing or stale.
- `RoomSpaceRoot`: content parented under it keeps room-registered local
  coordinates across sessions (`WaitForBindAsync`, `WorldToRoom`, `Adopt`,
  `AdoptAtRoomOrigin`).

### Game integration — `RoomScanSession`

- `StartScanAsync` → `FreezeInView` → `FinalizeScanAsync` → `ScanResult`;
  `LoadAsync` / `LoadLatestAsync` / `LoadRefinedOnlyAsync`;
  `ListSavedScans` / `DeleteScanAsync` / `UnloadActiveScanAsync` /
  `ClearAllScansAsync`; `RefinedMesh` / `RefinedAtlas` / `RefinedMeshRenderer`.
- One serialised permission queue (`AndroidRuntimePermission`): one dialog at
  a time, same-permission callers share a task. `StartScanAsync` requests
  what is missing before bring-up — `USE_SCENE` (required), then
  `HEADSET_CAMERA` / `USE_ANCHOR_API` (degrade). Hosts may front-load via
  `Request{Scene,Camera,Anchor}PermissionAsync` / `Has*Permission`.
- Scene-room occupancy from outer wall planes, never `GetCurrentRoom()`:
  `IsHeadsetInsideASceneRoom`, `IsHeadsetInsideBoundSceneRoom`,
  `HeadsetSceneRoomUuid`, `BoundSceneRoomUuid`,
  `TryRebindBoundSceneRoomIfHeadsetMatches`, `CopyHeadsetRoomWallFaces`
  (`SceneWallFace`, `IsScreen` for TVs).
- Discovery control: `WaitUntilRoomReadyAsync`, `IsRoomLoaded`,
  `HasSceneRooms`, `ReloadSceneFromDeviceAsync`,
  `RequestSpaceSetupAndReloadAsync`. `RoomReady` always signals (zero rooms,
  denied permission, missing `SceneLoadedEvent`); scene load never
  auto-launches Space Setup.
- Depth sensor and RGB camera run only during a scan; both are disabled at
  `Awake` and by the wizard.

### Beyond the mesh

- Gaussian Splat pipeline: keyframes + PLY point cloud → server training →
  on-device rendering (`HAS_GAUSSIAN_SPLATTING`).
- Optional YOLO object detection via Unity Inference Engine with GPU NMS
  (`HAS_AI_INFERENCE`), projected to 3D and merged with MRUK anchors in a
  `SceneObjectRegistry`; debug visualizer.
- Server-side atlas and mesh enhancement clients.

### Tooling

- Game-ready setup wizard: Meta Building Blocks, URP pipeline, build profile,
  VR UI input pipeline, `AndroidManifest`, shader wiring, xatlas build,
  scan defaults; turns `OVRManager`'s startup permission dialog off.
- Two-panel world-space VR debug menu: scan, saved-scan browser, refine,
  Gaussian Splat, tools.
- Meta XR SDK 205, Unity 6; every shader compiles warning-free on Metal and
  Vulkan.

## [0.1.0] - 2026-03-08

Initial tag: GPU TSDF volume integration from Quest depth, CPU mesh
extraction, triplanar world-space texture cache, plane detection and
4-phase mesh regularization, fast-convergence stability overhaul.
