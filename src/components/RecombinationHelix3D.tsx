import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { ChevronDown, ChevronUp, Maximize2, Minimize2, RotateCcw } from 'lucide-react';
import {
  STRESS_TEST_BREAKPOINTS,
  formatBp,
} from '../utils/genomeScan';
import {
  HELIX_CENTER_Y,
  applyExplodeOffset,
  applySceneTheme,
  bpToHelixX,
  buildRecombinationHelixScene,
  updateHelixBreakpointColors,
  type HelixSceneObjects,
} from './three/helixSceneBuilder';

type CameraPresetId = 'overview' | 'breakpoint' | 'top';

interface RecombinationHelix3DProps {
  selectedEventId: number;
  onSelectEvent: (eventId: number) => void;
  isDark: boolean;
}

const _camTargetPos = new THREE.Vector3();
const _orbitTargetPos = new THREE.Vector3();

function detectSoftwareWebGL(): boolean {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

export const RecombinationHelix3D: React.FC<RecombinationHelix3DProps> = ({
  selectedEventId,
  onSelectEvent,
  isDark,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [cameraPreset, setCameraPreset] = useState<CameraPresetId>('overview');
  const [explodePct, setExplodePct] = useState<number>(0);
  const [dockCollapsed, setDockCollapsed] = useState<boolean>(false);
  const [fullscreen, setFullscreen] = useState<boolean>(false);

  const selectedEvent =
    STRESS_TEST_BREAKPOINTS.find((ev) => ev.id === selectedEventId) ?? STRESS_TEST_BREAKPOINTS[6];

  // Mutable refs consumed by the Three.js loop without recreating WebGLRenderer
  const stateRef = useRef({
    selectedEvent,
    isDark,
    explodeTarget: 0,
    explodeCurrent: 0,
    caliperXTarget: bpToHelixX(selectedEvent.bp),
    caliperXCurrent: bpToHelixX(selectedEvent.bp),
    cameraPreset: 'overview' as CameraPresetId,
    transitionActive: false,
  });

  const engineRef = useRef<{
    scene: THREE.Scene;
    objects: HelixSceneObjects;
    keyLight: THREE.DirectionalLight;
    hemiLight: THREE.HemisphereLight;
    maxAnisotropy: number;
    requestRender: () => void;
    resize: () => void;
  } | null>(null);

  // Keyboard shortcut 'H' to toggle HUD dock cleanly
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if (e.key === 'h' || e.key === 'H') {
        setDockCollapsed((prev) => !prev);
      } else if (e.key === 'Escape' && fullscreen) {
        setFullscreen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [fullscreen]);

  // Mount Three.js WebGL scene once in stable ref container
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isSoftwareGl = detectSoftwareWebGL();
    const width = container.clientWidth || 960;
    const height = container.clientHeight || 540;

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const initialDark = stateRef.current.isDark;
    const bgHex = initialDark ? 0x121316 : 0xf2f2ee;
    scene.background = new THREE.Color(bgHex);
    scene.fog = new THREE.FogExp2(bgHex, 0.042);

    // Studio RoomEnvironment reflection map on every device
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    const envRT = pmremGenerator.fromScene(roomEnv, 0.04);
    scene.environment = envRT.texture;
    roomEnv.dispose();
    pmremGenerator.dispose();

    const initialAspect = width / height;
    const initialOverviewZ = initialAspect < 0.85 ? 13.2 : initialAspect < 1.1 ? 10.6 : 8.8;
    const camera = new THREE.PerspectiveCamera(40, initialAspect, 0.1, 60);
    camera.position.set(0.8, initialAspect < 0.85 ? 4.8 : 4.2, initialOverviewZ);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, HELIX_CENTER_Y - 0.1, 0);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.enablePan = false;
    controls.minDistance = 3.2;
    controls.maxDistance = 17.5;
    controls.minPolarAngle = 0.2;
    controls.maxPolarAngle = Math.PI * 0.49;

    // 3-point balanced studio lighting rig (warm key + cool hemisphere fill + rim)
    const keyLight = new THREE.DirectionalLight(initialDark ? 0xfff3e0 : 0xfff8eb, initialDark ? 2.5 : 2.3);
    keyLight.position.set(6.5, 9.5, 6.2);
    keyLight.castShadow = !isSoftwareGl;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 1;
    keyLight.shadow.camera.far = 26;
    const shadowBound = 6.5;
    keyLight.shadow.camera.left = -shadowBound;
    keyLight.shadow.camera.right = shadowBound;
    keyLight.shadow.camera.top = shadowBound;
    keyLight.shadow.camera.bottom = -shadowBound;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const hemiLight = new THREE.HemisphereLight(
      initialDark ? 0x6b8cff : 0xc7d8ff,
      initialDark ? 0x12151c : 0xe5e3db,
      initialDark ? 1.05 : 1.15
    );
    scene.add(hemiLight);

    const rimLight = new THREE.DirectionalLight(0x2dd4a4, 0.9);
    rimLight.position.set(-6.5, 4.5, -5.5);
    scene.add(rimLight);

    const maxAnisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    const objects = buildRecombinationHelixScene(
      maxAnisotropy,
      stateRef.current.selectedEvent,
      initialDark
    );
    scene.add(objects.rootGroup);

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    // Coalesced on-demand renderer with Software-WebGL fence pacing (zero thread starvation)
    let frameId = 0;
    let renderBusy = false;
    let renderDirty = false;
    let disposed = false;
    let lastTime = performance.now();

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      if (typeof gl.fenceSync !== 'function') {
        frameId = window.setTimeout(cb, 80);
        return;
      }
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) return;
        const status = sync ? gl.getSyncParameter(sync, gl.SYNC_STATUS) : gl.SIGNALED;
        if (status === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    function computePresetTargets(preset: CameraPresetId, bpX: number) {
      const aspect = camera.aspect;
      if (preset === 'breakpoint') {
        const zBp = aspect < 0.85 ? 5.2 : 3.85;
        _camTargetPos.set(bpX + 0.65, HELIX_CENTER_Y + 0.7, zBp);
        _orbitTargetPos.set(bpX, HELIX_CENTER_Y, 0);
      } else if (preset === 'top') {
        const yTop = aspect < 0.85 ? 13.2 : 10.6;
        _camTargetPos.set(0, yTop, 1.35);
        _orbitTargetPos.set(0, HELIX_CENTER_Y - 0.1, 0);
      } else {
        const zOverview = aspect < 0.85 ? 13.2 : aspect < 1.1 ? 10.6 : 8.8;
        const yOverview = aspect < 0.85 ? 4.8 : 4.2;
        _camTargetPos.set(0.8, yOverview, zOverview);
        _orbitTargetPos.set(0, HELIX_CENTER_Y - 0.1, 0);
      }
    }

    function requestRender() {
      if (disposed) return;
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = window.requestAnimationFrame((now) => {
        if (disposed) return;
        renderDirty = false;
        const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000));
        lastTime = now;
        const st = stateRef.current;

        let needsMoreFrames = false;

        // Smoothly damp caliper X coordinate to active breakpoint
        if (Math.abs(st.caliperXCurrent - st.caliperXTarget) > 0.004) {
          st.caliperXCurrent = isSoftwareGl
            ? st.caliperXTarget
            : THREE.MathUtils.damp(st.caliperXCurrent, st.caliperXTarget, 8, dt);
          objects.caliperGroup.position.x = st.caliperXCurrent;
          needsMoreFrames = !isSoftwareGl;
        } else {
          objects.caliperGroup.position.x = st.caliperXTarget;
        }

        // Smoothly damp exploded strand separation
        if (Math.abs(st.explodeCurrent - st.explodeTarget) > 0.003) {
          st.explodeCurrent = isSoftwareGl
            ? st.explodeTarget
            : THREE.MathUtils.damp(st.explodeCurrent, st.explodeTarget, 9, dt);
          applyExplodeOffset(objects, st.explodeCurrent);
          needsMoreFrames = !isSoftwareGl;
        }

        // Smoothly damp scripted camera preset transitions (cancelled if user interacts)
        if (st.transitionActive) {
          computePresetTargets(st.cameraPreset, st.caliperXTarget);
          if (isSoftwareGl) {
            camera.position.copy(_camTargetPos);
            controls.target.copy(_orbitTargetPos);
            st.transitionActive = false;
          } else {
            const alpha = 1 - Math.exp(-7 * dt);
            camera.position.lerp(_camTargetPos, alpha);
            controls.target.lerp(_orbitTargetPos, alpha);
            if (
              camera.position.distanceToSquared(_camTargetPos) < 0.0008 &&
              controls.target.distanceToSquared(_orbitTargetPos) < 0.0008
            ) {
              camera.position.copy(_camTargetPos);
              controls.target.copy(_orbitTargetPos);
              st.transitionActive = false;
            } else {
              needsMoreFrames = true;
            }
          }
        }

        const controlsMoved = controls.update();
        if (controlsMoved && !isSoftwareGl) {
          needsMoreFrames = true;
        }

        renderer.render(scene, camera);

        const done = () => {
          renderBusy = false;
          if (!disposed && (renderDirty || needsMoreFrames)) {
            requestRender();
          }
        };
        if (isSoftwareGl) afterDraw(done);
        else done();
      });
    }

    const handleControlStart = () => {
      stateRef.current.transitionActive = false;
    };
    const handleControlChange = () => {
      requestRender();
    };
    controls.addEventListener('start', handleControlStart);
    controls.addEventListener('change', handleControlChange);

    const handleResize = () => {
      if (!container || disposed) return;
      const w = container.clientWidth || 640;
      const h = container.clientHeight || 420;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      requestRender();
    };

    const resizeObserver =
      typeof ResizeObserver !== 'undefined' ? new ResizeObserver(handleResize) : null;
    resizeObserver?.observe(container);
    window.addEventListener('resize', handleResize);

    engineRef.current = {
      scene,
      objects,
      keyLight,
      hemiLight,
      maxAnisotropy,
      requestRender,
      resize: handleResize,
    };

    // Initial render
    requestRender();

    return () => {
      disposed = true;
      window.cancelAnimationFrame(frameId);
      window.clearTimeout(frameId);
      resizeObserver?.disconnect();
      window.removeEventListener('resize', handleResize);
      controls.removeEventListener('start', handleControlStart);
      controls.removeEventListener('change', handleControlChange);
      controls.dispose();
      objects.geometries.forEach((g) => g.dispose());
      Object.values(objects.materials).forEach((m) => m.dispose());
      objects.textures.forEach((t) => t.dispose());
      envRT.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      engineRef.current = null;
    };
  }, []);

  // Sync theme changes live into Three.js studio lighting & fog
  useEffect(() => {
    stateRef.current.isDark = isDark;
    const engine = engineRef.current;
    if (!engine) return;
    applySceneTheme(
      engine.scene,
      engine.objects,
      engine.keyLight,
      engine.hemiLight,
      engine.maxAnisotropy,
      isDark,
      stateRef.current.selectedEvent
    );
    engine.requestRender();
  }, [isDark]);

  // Sync active breakpoint coordinate & clade colors
  useEffect(() => {
    stateRef.current.selectedEvent = selectedEvent;
    stateRef.current.caliperXTarget = bpToHelixX(selectedEvent.bp);
    const engine = engineRef.current;
    if (!engine) return;
    updateHelixBreakpointColors(engine.objects, selectedEvent, stateRef.current.isDark);
    if (stateRef.current.cameraPreset === 'breakpoint') {
      stateRef.current.transitionActive = true;
    }
    engine.requestRender();
  }, [selectedEvent]);

  // Sync explode slider
  useEffect(() => {
    stateRef.current.explodeTarget = explodePct / 100;
    engineRef.current?.requestRender();
  }, [explodePct]);

  // Resize renderer smoothly when toggling fullscreen viewport mode
  useEffect(() => {
    const id = window.setTimeout(() => {
      engineRef.current?.resize();
    }, 30);
    return () => window.clearTimeout(id);
  }, [fullscreen]);

  const handleSelectCameraPreset = (preset: CameraPresetId) => {
    setCameraPreset(preset);
    stateRef.current.cameraPreset = preset;
    stateRef.current.transitionActive = true;
    engineRef.current?.requestRender();
  };

  const handleResetView = () => {
    setExplodePct(0);
    handleSelectCameraPreset('overview');
  };

  return (
    <section
      id="explorer-3d"
      aria-labelledby="explorer-3d-heading"
      className={
        fullscreen
          ? 'fixed inset-0 z-50 bg-[#F2F2EE] dark:bg-[#121316]'
          : 'scroll-mt-16 bg-[#F2F2EE] dark:bg-[#121316] border-t border-[#E6E5DF] dark:border-[#2B2E36]'
      }
    >
      <div
        className={`relative w-full overflow-hidden ${
          fullscreen ? 'h-screen w-screen' : 'h-[560px] sm:h-[620px]'
        }`}
      >
        {/* Dedicated empty WebGL mount ref outside sibling React HUD */}
        <div
          ref={mountRef}
          data-testid="threejs-helix-canvas"
          className="absolute inset-0 w-full h-full cursor-grab active:cursor-grabbing"
          aria-label="Interactive 3D viral recombination double-helix and breakpoint caliper model"
        />

        {/* Top HUD Row: Spec Badge + Fullscreen Toggle (unified flex row prevents mobile overlap; zero backdrop-filter) */}
        <div className="pointer-events-none absolute inset-x-4 top-4 sm:inset-x-6 sm:top-6 z-10 flex items-start justify-between gap-2.5">
          <div className="pointer-events-auto min-w-0 max-w-[360px] rounded-xl border border-[#DFDED8] dark:border-[#2B2E36] bg-[rgba(255,255,255,0.92)] dark:bg-[rgba(22,24,29,0.92)] px-3.5 py-2.5 sm:px-4 sm:py-3 shadow-sm">
            <p className="font-mono text-[10px] sm:text-[10.5px] uppercase tracking-[0.11em] text-[#1B9E77] dark:text-[#2DD4A4] truncate">
              3D Molecular Crossover · Three.js PBR
            </p>
            <h2
              id="explorer-3d-heading"
              className="mt-0.5 sm:mt-1 text-[15px] sm:text-[17px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]"
            >
              {formatBp(selectedEvent.bp)} · {selectedEvent.parents}
            </h2>
            <p className="mt-0.5 font-mono text-[10.5px] sm:text-[11px] text-[#5A5A62] dark:text-[#A8ADB8] tabular-nums">
              {selectedEvent.carriers} · z = {selectedEvent.zScore.toFixed(1)} · Drag to orbit
            </p>
          </div>

          <button
            type="button"
            onClick={() => setFullscreen((f) => !f)}
            aria-label={fullscreen ? 'Exit fullscreen 3D explorer' : 'Expand 3D explorer to fullscreen'}
            className="pointer-events-auto shrink-0 inline-flex items-center gap-1.5 rounded-lg border border-[#DFDED8] dark:border-[#2B2E36] bg-[rgba(255,255,255,0.92)] dark:bg-[rgba(22,24,29,0.92)] px-2.5 py-2 sm:px-3 text-[12px] font-mono text-[#141416] dark:text-[#F3F4F6] hover:border-[#1B9E77] dark:hover:border-[#2DD4A4] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            {fullscreen ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Exit 3D</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5" aria-hidden="true" />
                <span className="hidden sm:inline">Fullscreen</span>
              </>
            )}
          </button>
        </div>

        {/* Unified Collapsible Bottom Control Dock (Zero Viewport Occlusion, no backdrop-filter) */}
        <div
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 w-[min(860px,94%)] flex flex-col items-center"
          data-testid="threejs-bottom-dock"
        >
          <button
            type="button"
            onClick={() => setDockCollapsed((c) => !c)}
            aria-expanded={!dockCollapsed}
            className="mb-1.5 inline-flex items-center gap-1.5 rounded-full border border-[#DFDED8] dark:border-[#2B2E36] bg-[rgba(255,255,255,0.94)] dark:bg-[rgba(22,24,29,0.94)] px-3 py-1 text-[11px] font-mono text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-white transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            {dockCollapsed ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Show 3D Controls (H)</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Hide Controls (H)</span>
              </>
            )}
          </button>

          {!dockCollapsed && (
            <div className="w-full rounded-2xl border border-[#DFDED8] dark:border-[#2B2E36] bg-[rgba(255,255,255,0.95)] dark:bg-[rgba(22,24,29,0.95)] px-4 py-3 shadow-lg flex flex-wrap items-center justify-between gap-3">
              {/* Camera Presets */}
              <div className="flex items-center gap-1.5" role="group" aria-label="3D camera views">
                {(
                  [
                    { id: 'overview', label: 'Overview' },
                    { id: 'breakpoint', label: 'Breakpoint' },
                    { id: 'top', label: 'Top' },
                  ] as const
                ).map((preset) => {
                  const active = cameraPreset === preset.id;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => handleSelectCameraPreset(preset.id)}
                      aria-pressed={active}
                      className={`px-2.5 py-1 text-[11.5px] font-mono rounded-lg transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[#4F6BF6] ${
                        active
                          ? 'bg-[#141416] dark:bg-[#2DD4A4] text-white dark:text-[#0E1013] font-semibold'
                          : 'bg-[#F6F6F3] dark:bg-[#1E2128] text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-white'
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>

              {/* Exploded Strand Separation Slider */}
              <label className="flex items-center gap-2.5 text-[11.5px] font-mono text-[#5A5A62] dark:text-[#A8ADB8]">
                <span className="whitespace-nowrap">Strand Split</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={explodePct}
                  onChange={(e) => setExplodePct(Number(e.target.value))}
                  aria-label="Strand separation exploded view"
                  className="w-24 sm:w-28 accent-[#1B9E77] dark:accent-[#2DD4A4] cursor-pointer"
                />
                <span className="w-8 text-right tabular-nums text-[#141416] dark:text-[#F3F4F6]">
                  {explodePct}%
                </span>
              </label>

              {/* Breakpoint Quick Selector synced to Hero */}
              <div className="flex items-center gap-1.5">
                <span className="hidden md:inline font-mono text-[11px] text-[#5A5A62] dark:text-[#A8ADB8]">
                  Locus:
                </span>
                {[1, 4, 6, 7, 10].map((id) => {
                  const active = selectedEvent.id === id;
                  return (
                    <button
                      key={id}
                      type="button"
                      onClick={() => onSelectEvent(id)}
                      aria-pressed={active}
                      className={`px-2 py-0.5 text-[11px] font-mono rounded-md transition-colors tabular-nums focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[#4F6BF6] ${
                        active
                          ? 'bg-[#1B9E77] dark:bg-[#2DD4A4] text-white dark:text-[#0E1013] font-semibold'
                          : 'bg-[#F6F6F3] dark:bg-[#1E2128] text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-white'
                      }`}
                    >
                      #{id}
                    </button>
                  );
                })}
                <button
                  type="button"
                  onClick={handleResetView}
                  aria-label="Reset 3D view"
                  title="Reset 3D view"
                  className="ml-1 inline-flex items-center justify-center w-7 h-7 rounded-lg bg-[#F6F6F3] dark:bg-[#1E2128] text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-white transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
