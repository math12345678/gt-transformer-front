import * as THREE from 'three';
import {
  CLADE_COLORS,
  GENOME_LENGTH_BP,
  type BreakpointEvent,
} from '../../utils/genomeScan';

export interface HelixSceneObjects {
  rootGroup: THREE.Group;
  helixGroup: THREE.Group;
  strandALeftGroup: THREE.Group;
  strandARightGroup: THREE.Group;
  strandBGroup: THREE.Group;
  rungsMesh: THREE.InstancedMesh;
  nodesMesh: THREE.InstancedMesh;
  caliperGroup: THREE.Group;
  plinthMesh: THREE.Mesh;
  dialMesh: THREE.Mesh;
  shadowPlane: THREE.Mesh;
  materials: {
    leftBackbone: THREE.MeshStandardMaterial;
    rightBackbone: THREE.MeshStandardMaterial;
    compBackbone: THREE.MeshStandardMaterial;
    rungMat: THREE.MeshStandardMaterial;
    nodeMat: THREE.MeshStandardMaterial;
    caliperMetal: THREE.MeshStandardMaterial;
    caliperAccent: THREE.MeshStandardMaterial;
    flankLeftMat: THREE.MeshStandardMaterial;
    flankRightMat: THREE.MeshStandardMaterial;
    plinthMat: THREE.MeshStandardMaterial;
    dialMat: THREE.MeshStandardMaterial;
    shadowMat: THREE.MeshBasicMaterial;
  };
  rungBaseMatrices: THREE.Matrix4[];
  nodeBasePositions: { pos: THREE.Vector3; strand: 0 | 1 }[];
  textures: THREE.Texture[];
  geometries: THREE.BufferGeometry[];
}

export const HELIX_LENGTH = 8.4;
export const HELIX_RADIUS = 1.08;
export const HELIX_TURNS = 3.2;
export const HELIX_CENTER_Y = 2.05;
const RUNG_COUNT = 56;

const _dummy = new THREE.Object3D();
const _vecA = new THREE.Vector3();
const _vecB = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _quat = new THREE.Quaternion();
const _color = new THREE.Color();

export function bpToHelixX(bp: number): number {
  const t = THREE.MathUtils.clamp(bp / GENOME_LENGTH_BP, 0.04, 0.96);
  return (t - 0.5) * HELIX_LENGTH;
}

function helixPoint(t: number, phaseOffset: number, out: THREE.Vector3): THREE.Vector3 {
  const x = (t - 0.5) * HELIX_LENGTH;
  const angle = t * Math.PI * 2 * HELIX_TURNS + phaseOffset;
  const y = HELIX_CENTER_Y + Math.sin(angle) * HELIX_RADIUS;
  const z = Math.cos(angle) * HELIX_RADIUS;
  return out.set(x, y, z);
}

/**
 * Generates a crisp procedural calibrated genome dial texture on an HTML5 canvas
 * (no external asset URLs, SRGBColorSpace + max anisotropy).
 */
function createCoordinateDialTexture(maxAnisotropy: number, isDark: boolean): THREE.CanvasTexture {
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.clearRect(0, 0, size, size);
    const cx = size / 2;
    const cy = size / 2;
    const outerR = size * 0.46;
    const innerR = size * 0.34;

    ctx.strokeStyle = isDark ? 'rgba(45, 212, 164, 0.35)' : 'rgba(27, 158, 119, 0.38)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(20, 20, 22, 0.14)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
    ctx.stroke();

    // 72 minor ticks + 12 major kb ticks
    for (let i = 0; i < 72; i++) {
      const a = (i / 72) * Math.PI * 2 - Math.PI / 2;
      const isMajor = i % 6 === 0;
      const r1 = outerR - (isMajor ? 28 : 14);
      const r2 = outerR - 2;
      ctx.strokeStyle = isMajor
        ? isDark
          ? 'rgba(45, 212, 164, 0.85)'
          : 'rgba(27, 158, 119, 0.88)'
        : isDark
          ? 'rgba(255, 255, 255, 0.25)'
          : 'rgba(20, 20, 22, 0.22)';
      ctx.lineWidth = isMajor ? 4 : 2;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
      ctx.stroke();

      if (isMajor) {
        const kb = i / 6;
        const labelR = innerR + 26;
        ctx.save();
        ctx.translate(cx + Math.cos(a) * labelR, cy + Math.sin(a) * labelR);
        ctx.rotate(a + Math.PI / 2);
        ctx.fillStyle = isDark ? 'rgba(243, 244, 246, 0.72)' : 'rgba(20, 20, 22, 0.65)';
        ctx.font = '600 24px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${kb}kb`, 0, 0);
        ctx.restore();
      }
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = maxAnisotropy;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Generates a soft radial contact shadow texture for grounding the specimen plinth.
 */
function createContactShadowTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    const grad = ctx.createRadialGradient(
      size / 2,
      size / 2,
      size * 0.08,
      size / 2,
      size / 2,
      size * 0.5
    );
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.55)');
    grad.addColorStop(0.55, 'rgba(0, 0, 0, 0.18)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Generates subtle micro-brushed roughness map on canvas for metallic PBR variation.
 */
function createBrushedRoughnessTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, size, size);
    for (let y = 0; y < size; y++) {
      const shade = 112 + Math.floor(Math.sin(y * 1.9) * 18 + Math.cos(y * 0.7) * 12);
      ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
      ctx.fillRect(0, y, size, 1);
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 2);
  tex.needsUpdate = true;
  return tex;
}

export function buildRecombinationHelixScene(
  maxAnisotropy: number,
  initialEvent: BreakpointEvent,
  isDark: boolean
): HelixSceneObjects {
  const geometries: THREE.BufferGeometry[] = [];
  const textures: THREE.Texture[] = [];

  const brushedRoughness = createBrushedRoughnessTexture();
  const dialTex = createCoordinateDialTexture(maxAnisotropy, isDark);
  const shadowTex = createContactShadowTexture();
  textures.push(brushedRoughness, dialTex, shadowTex);

  // Materials with varied roughness (>= 0.25 contrast across adjacent parts) and metalness
  const leftBackbone = new THREE.MeshStandardMaterial({
    color: isDark ? 0x94a3b8 : 0x262930,
    roughness: 0.26,
    metalness: 0.76,
    roughnessMap: brushedRoughness,
  });

  const rightBackbone = new THREE.MeshStandardMaterial({
    color: isDark ? 0x2dd4a4 : 0x1b9e77,
    roughness: 0.34,
    metalness: 0.28,
    emissive: isDark ? 0x0b3d30 : 0x062920,
    emissiveIntensity: isDark ? 0.35 : 0.14,
  });

  const compBackbone = new THREE.MeshStandardMaterial({
    color: isDark ? 0x7c93ff : 0x4f6bf6,
    roughness: 0.42,
    metalness: 0.24,
    emissive: isDark ? 0x162259 : 0x0e1842,
    emissiveIntensity: isDark ? 0.28 : 0.1,
  });

  const rungMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.56,
    metalness: 0.12,
  });

  const nodeMat = new THREE.MeshStandardMaterial({
    color: isDark ? 0xe2e8f0 : 0xfafaf8,
    roughness: 0.16,
    metalness: 0.86,
  });

  const caliperMetal = new THREE.MeshStandardMaterial({
    color: isDark ? 0x3b404d : 0x1e2026,
    roughness: 0.22,
    metalness: 0.85,
  });

  const caliperAccent = new THREE.MeshStandardMaterial({
    color: isDark ? 0x34d399 : 0x1b9e77,
    roughness: 0.2,
    metalness: 0.35,
    emissive: isDark ? 0x1b9e77 : 0x116b51,
    emissiveIntensity: isDark ? 0.8 : 0.35,
  });

  const flankLeftMat = new THREE.MeshStandardMaterial({
    color: isDark ? 0x94a3b8 : 0x262930,
    roughness: 0.4,
    metalness: 0.4,
    transparent: true,
    opacity: 0.34,
  });

  const flankRightMat = new THREE.MeshStandardMaterial({
    color: isDark ? 0x7c93ff : 0x4f6bf6,
    roughness: 0.4,
    metalness: 0.4,
    transparent: true,
    opacity: 0.38,
  });

  const plinthMat = new THREE.MeshStandardMaterial({
    color: isDark ? 0x1e2128 : 0xe9e8e2,
    roughness: 0.62,
    metalness: 0.22,
    roughnessMap: brushedRoughness,
  });

  const dialMat = new THREE.MeshStandardMaterial({
    map: dialTex,
    transparent: true,
    roughness: 0.45,
    metalness: 0.1,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });

  const shadowMat = new THREE.MeshBasicMaterial({
    map: shadowTex,
    transparent: true,
    depthWrite: false,
    opacity: isDark ? 0.55 : 0.36,
  });

  const rootGroup = new THREE.Group();
  const helixGroup = new THREE.Group();
  rootGroup.add(helixGroup);

  // 1. Grounded Optical Specimen Stage Plinth (LatheGeometry, bottom-center flush at y = 0)
  const plinthProfile = [
    new THREE.Vector2(0.0, 0.0),
    new THREE.Vector2(4.95, 0.0),
    new THREE.Vector2(5.05, 0.06),
    new THREE.Vector2(4.78, 0.16),
    new THREE.Vector2(4.35, 0.19),
    new THREE.Vector2(4.25, 0.26),
    new THREE.Vector2(0.0, 0.26),
  ];
  const plinthGeo = new THREE.LatheGeometry(plinthProfile, 64);
  geometries.push(plinthGeo);
  const plinthMesh = new THREE.Mesh(plinthGeo, plinthMat);
  plinthMesh.receiveShadow = true;
  plinthMesh.castShadow = false;
  rootGroup.add(plinthMesh);

  // 2. Calibrated Genome Coordinate Dial Ring on top of plinth (offset +0.02 above plinth top)
  const dialGeo = new THREE.CircleGeometry(4.15, 64);
  dialGeo.rotateX(-Math.PI / 2);
  geometries.push(dialGeo);
  const dialMesh = new THREE.Mesh(dialGeo, dialMat);
  dialMesh.position.y = 0.28;
  rootGroup.add(dialMesh);

  // 3. Soft Radial Contact Shadow Plane grounded beneath the plinth
  const shadowGeo = new THREE.PlaneGeometry(12.2, 12.2);
  shadowGeo.rotateX(-Math.PI / 2);
  geometries.push(shadowGeo);
  const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
  shadowPlane.position.y = 0.005;
  rootGroup.add(shadowPlane);

  // 4. Support Pylons anchoring the helix ends to the stage plinth (LatheGeometry)
  const pylonProfile = [
    new THREE.Vector2(0.0, 0.0),
    new THREE.Vector2(0.24, 0.0),
    new THREE.Vector2(0.18, 0.16),
    new THREE.Vector2(0.08, 0.38),
    new THREE.Vector2(0.06, 1.56),
    new THREE.Vector2(0.14, 1.72),
    new THREE.Vector2(0.0, 1.78),
  ];
  const pylonGeo = new THREE.LatheGeometry(pylonProfile, 24);
  geometries.push(pylonGeo);
  for (const sign of [-1, 1]) {
    const pylon = new THREE.Mesh(pylonGeo, caliperMetal);
    pylon.position.set(sign * (HELIX_LENGTH * 0.5 + 0.12), 0.26, 0);
    pylon.castShadow = true;
    pylon.receiveShadow = true;
    rootGroup.add(pylon);
  }

  // 5. Helical Strand A (split at mid-crossover) and Strand B via TubeGeometry
  const splitT = initialEvent.bp / GENOME_LENGTH_BP;
  const strandALeftPts: THREE.Vector3[] = [];
  const strandARightPts: THREE.Vector3[] = [];
  const strandBPts: THREE.Vector3[] = [];
  const samples = 96;
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const pA = helixPoint(t, 0, new THREE.Vector3());
    const pB = helixPoint(t, Math.PI, new THREE.Vector3());
    if (t <= splitT + 0.015) strandALeftPts.push(pA.clone());
    if (t >= splitT - 0.015) strandARightPts.push(pA.clone());
    strandBPts.push(pB);
  }

  const strandALeftGroup = new THREE.Group();
  const strandARightGroup = new THREE.Group();
  const strandBGroup = new THREE.Group();

  const tubeALeftGeo = new THREE.TubeGeometry(
    new THREE.CatmullRomCurve3(strandALeftPts),
    72,
    0.092,
    14,
    false
  );
  const tubeARightGeo = new THREE.TubeGeometry(
    new THREE.CatmullRomCurve3(strandARightPts),
    48,
    0.096,
    14,
    false
  );
  const tubeBGeo = new THREE.TubeGeometry(
    new THREE.CatmullRomCurve3(strandBPts),
    96,
    0.088,
    14,
    false
  );
  geometries.push(tubeALeftGeo, tubeARightGeo, tubeBGeo);

  const meshALeft = new THREE.Mesh(tubeALeftGeo, leftBackbone);
  meshALeft.castShadow = true;
  meshALeft.receiveShadow = true;
  strandALeftGroup.add(meshALeft);

  const meshARight = new THREE.Mesh(tubeARightGeo, rightBackbone);
  meshARight.castShadow = true;
  meshARight.receiveShadow = true;
  strandARightGroup.add(meshARight);

  const meshB = new THREE.Mesh(tubeBGeo, compBackbone);
  meshB.castShadow = true;
  meshB.receiveShadow = true;
  strandBGroup.add(meshB);

  helixGroup.add(strandALeftGroup, strandARightGroup, strandBGroup);

  // 6. Instanced Base-Pair Rung Bridges (56 instances colored by parental clade tracts)
  const rungGeo = new THREE.CylinderGeometry(0.036, 0.036, 1.0, 10, 1);
  geometries.push(rungGeo);
  const rungsMesh = new THREE.InstancedMesh(rungGeo, rungMat, RUNG_COUNT);
  rungsMesh.castShadow = true;
  rungsMesh.receiveShadow = true;

  const rungBaseMatrices: THREE.Matrix4[] = [];
  const leftHex = CLADE_COLORS[initialEvent.leftParent].fill;
  const rightHex = CLADE_COLORS[initialEvent.rightParent].fill;

  for (let i = 0; i < RUNG_COUNT; i++) {
    const t = (i + 0.5) / RUNG_COUNT;
    helixPoint(t, 0, _vecA);
    helixPoint(t, Math.PI, _vecB);
    _mid.copy(_vecA).add(_vecB).multiplyScalar(0.5);
    _dir.copy(_vecB).sub(_vecA);
    const len = _dir.length();
    _dir.normalize();
    _quat.setFromUnitVectors(_up, _dir);

    _dummy.position.copy(_mid);
    _dummy.quaternion.copy(_quat);
    _dummy.scale.set(1, len * 0.92, 1);
    _dummy.updateMatrix();

    rungsMesh.setMatrixAt(i, _dummy.matrix);
    rungBaseMatrices.push(_dummy.matrix.clone());

    const isDonor = t >= splitT;
    _color.set(isDonor ? rightHex : leftHex);
    // Subtle organic variation per rung
    _color.offsetHSL(0, 0, ((i % 3) - 1) * 0.04);
    rungsMesh.setColorAt(i, _color);
  }
  rungsMesh.instanceMatrix.needsUpdate = true;
  if (rungsMesh.instanceColor) rungsMesh.instanceColor.needsUpdate = true;
  rungsMesh.computeBoundingSphere();
  rungsMesh.frustumCulled = false;
  helixGroup.add(rungsMesh);

  // 7. Instanced Phosphate Backbone Nodes (112 spheres)
  const nodeGeo = new THREE.SphereGeometry(0.115, 14, 12);
  geometries.push(nodeGeo);
  const nodesMesh = new THREE.InstancedMesh(nodeGeo, nodeMat, RUNG_COUNT * 2);
  nodesMesh.castShadow = true;

  const nodeBasePositions: { pos: THREE.Vector3; strand: 0 | 1 }[] = [];
  for (let i = 0; i < RUNG_COUNT; i++) {
    const t = (i + 0.5) / RUNG_COUNT;
    const pA = helixPoint(t, 0, new THREE.Vector3());
    const pB = helixPoint(t, Math.PI, new THREE.Vector3());

    const scaleJitter = 0.92 + ((i * 7) % 5) * 0.04;
    _dummy.position.copy(pA);
    _dummy.quaternion.identity();
    _dummy.scale.setScalar(scaleJitter);
    _dummy.updateMatrix();
    nodesMesh.setMatrixAt(i * 2, _dummy.matrix);
    nodeBasePositions.push({ pos: pA, strand: 0 });

    _dummy.position.copy(pB);
    _dummy.scale.setScalar(scaleJitter);
    _dummy.updateMatrix();
    nodesMesh.setMatrixAt(i * 2 + 1, _dummy.matrix);
    nodeBasePositions.push({ pos: pB, strand: 1 });
  }
  nodesMesh.instanceMatrix.needsUpdate = true;
  nodesMesh.computeBoundingSphere();
  nodesMesh.frustumCulled = false;
  helixGroup.add(nodesMesh);

  // 8. Composite Machined Breakpoint Caliper Ring Assembly (slides to active breakpoint!)
  const caliperGroup = new THREE.Group();
  const caliperOuterGeo = new THREE.TorusGeometry(HELIX_RADIUS + 0.34, 0.055, 16, 64);
  caliperOuterGeo.rotateY(Math.PI / 2);
  const caliperInnerGeo = new THREE.TorusGeometry(HELIX_RADIUS + 0.25, 0.026, 12, 64);
  caliperInnerGeo.rotateY(Math.PI / 2);
  const flankBandGeo = new THREE.CylinderGeometry(
    HELIX_RADIUS + 0.22,
    HELIX_RADIUS + 0.22,
    0.52,
    36,
    1,
    true
  );
  flankBandGeo.rotateZ(Math.PI / 2);
  geometries.push(caliperOuterGeo, caliperInnerGeo, flankBandGeo);

  const caliperOuter = new THREE.Mesh(caliperOuterGeo, caliperMetal);
  caliperOuter.castShadow = true;
  const caliperInner = new THREE.Mesh(caliperInnerGeo, caliperAccent);

  const flankLeftBand = new THREE.Mesh(flankBandGeo, flankLeftMat);
  flankLeftBand.position.x = -0.34;
  const flankRightBand = new THREE.Mesh(flankBandGeo, flankRightMat);
  flankRightBand.position.x = 0.34;

  // 4 precision tick pins around the caliper ring
  const pinGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.22, 8);
  geometries.push(pinGeo);
  for (let p = 0; p < 4; p++) {
    const a = (p / 4) * Math.PI * 2;
    const pin = new THREE.Mesh(pinGeo, caliperAccent);
    pin.position.set(0, Math.sin(a) * (HELIX_RADIUS + 0.42), Math.cos(a) * (HELIX_RADIUS + 0.42));
    pin.rotation.x = a;
    caliperGroup.add(pin);
  }

  caliperGroup.add(caliperOuter, caliperInner, flankLeftBand, flankRightBand);
  caliperGroup.position.set(bpToHelixX(initialEvent.bp), HELIX_CENTER_Y, 0);
  helixGroup.add(caliperGroup);

  return {
    rootGroup,
    helixGroup,
    strandALeftGroup,
    strandARightGroup,
    strandBGroup,
    rungsMesh,
    nodesMesh,
    caliperGroup,
    plinthMesh,
    dialMesh,
    shadowPlane,
    materials: {
      leftBackbone,
      rightBackbone,
      compBackbone,
      rungMat,
      nodeMat,
      caliperMetal,
      caliperAccent,
      flankLeftMat,
      flankRightMat,
      plinthMat,
      dialMat,
      shadowMat,
    },
    rungBaseMatrices,
    nodeBasePositions,
    textures,
    geometries,
  };
}

export function updateHelixBreakpointColors(
  objects: HelixSceneObjects,
  event: BreakpointEvent,
  isDark: boolean
) {
  const splitT = event.bp / GENOME_LENGTH_BP;
  const leftHex = isDark
    ? event.leftParent === 'CladeA'
      ? '#94A3B8'
      : CLADE_COLORS[event.leftParent].fill
    : CLADE_COLORS[event.leftParent].fill;
  const rightHex =
    isDark && event.rightParent === 'CladeB'
      ? '#2DD4A4'
      : CLADE_COLORS[event.rightParent].fill;

  for (let i = 0; i < RUNG_COUNT; i++) {
    const t = (i + 0.5) / RUNG_COUNT;
    _color.set(t >= splitT ? rightHex : leftHex);
    _color.offsetHSL(0, 0, ((i % 3) - 1) * 0.04);
    objects.rungsMesh.setColorAt(i, _color);
  }
  if (objects.rungsMesh.instanceColor) {
    objects.rungsMesh.instanceColor.needsUpdate = true;
  }
}

export function applyExplodeOffset(objects: HelixSceneObjects, explodeFactor: number) {
  const spread = explodeFactor * 0.65;
  objects.strandALeftGroup.position.y = spread;
  objects.strandARightGroup.position.y = spread;
  objects.strandBGroup.position.y = -spread;

  // Slightly scale rungs horizontally as strands separate so geometry stays visually coherent
  for (let i = 0; i < objects.rungBaseMatrices.length; i++) {
    _dummy.matrix.copy(objects.rungBaseMatrices[i]);
    _dummy.matrix.decompose(_dummy.position, _dummy.quaternion, _dummy.scale);
    _dummy.scale.y *= Math.max(0.28, 1 - explodeFactor * 0.55);
    _dummy.updateMatrix();
    objects.rungsMesh.setMatrixAt(i, _dummy.matrix);
  }
  objects.rungsMesh.instanceMatrix.needsUpdate = true;

  for (let i = 0; i < objects.nodeBasePositions.length; i++) {
    const entry = objects.nodeBasePositions[i];
    _dummy.position.copy(entry.pos);
    _dummy.position.y += entry.strand === 0 ? spread : -spread;
    _dummy.quaternion.identity();
    const scaleJitter = 0.92 + (((i >> 1) * 7) % 5) * 0.04;
    _dummy.scale.setScalar(scaleJitter);
    _dummy.updateMatrix();
    objects.nodesMesh.setMatrixAt(i, _dummy.matrix);
  }
  objects.nodesMesh.instanceMatrix.needsUpdate = true;
}

export function applySceneTheme(
  scene: THREE.Scene,
  objects: HelixSceneObjects,
  keyLight: THREE.DirectionalLight,
  hemiLight: THREE.HemisphereLight,
  maxAnisotropy: number,
  isDark: boolean,
  activeEvent: BreakpointEvent
) {
  const bgColor = isDark ? 0x121316 : 0xf2f2ee;
  scene.background = new THREE.Color(bgColor);
  scene.fog = new THREE.FogExp2(bgColor, 0.042);

  keyLight.color.setHex(isDark ? 0xfff3e0 : 0xfff8eb);
  keyLight.intensity = isDark ? 2.5 : 2.3;
  hemiLight.color.setHex(isDark ? 0x6b8cff : 0xc7d8ff);
  hemiLight.groundColor.setHex(isDark ? 0x12151c : 0xe5e3db);
  hemiLight.intensity = isDark ? 1.05 : 1.15;

  const { materials } = objects;
  materials.leftBackbone.color.setHex(isDark ? 0x94a3b8 : 0x262930);
  materials.rightBackbone.color.setHex(isDark ? 0x2dd4a4 : 0x1b9e77);
  materials.rightBackbone.emissive.setHex(isDark ? 0x0b3d30 : 0x062920);
  materials.rightBackbone.emissiveIntensity = isDark ? 0.35 : 0.14;
  materials.compBackbone.color.setHex(isDark ? 0x7c93ff : 0x4f6bf6);
  materials.nodeMat.color.setHex(isDark ? 0xe2e8f0 : 0xfafaf8);
  materials.caliperMetal.color.setHex(isDark ? 0x3b404d : 0x1e2026);
  materials.caliperAccent.color.setHex(isDark ? 0x34d399 : 0x1b9e77);
  materials.caliperAccent.emissiveIntensity = isDark ? 0.8 : 0.35;
  materials.plinthMat.color.setHex(isDark ? 0x1e2128 : 0xe9e8e2);
  materials.shadowMat.opacity = isDark ? 0.55 : 0.36;

  // Refresh coordinate dial texture for optimal contrast in current theme
  const nextDialTex = createCoordinateDialTexture(maxAnisotropy, isDark);
  const oldMap = materials.dialMat.map;
  materials.dialMat.map = nextDialTex;
  materials.dialMat.needsUpdate = true;
  if (oldMap) {
    oldMap.dispose();
  }
  objects.textures.push(nextDialTex);

  updateHelixBreakpointColors(objects, activeEvent, isDark);
}
