"use client";

import { PerformanceMonitor } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import * as THREE from "three";
import { Mark } from "@/components/ui/Logo";

/*
 * A big, diffuse cloud (the general-purpose model) continuously distills into a small, dense,
 * bright core (the specialist). Every particle's path lives in the vertex shader, so the CPU only
 * advances uniforms. Scrolling the hero collapses the whole cloud into the core, which is the one
 * thing in the scene bright enough to bloom.
 *
 * Without WebGL, or once the context is lost, the same ground is drawn as a still: ink, a soft
 * highlight where the core would sit, and the mark. A machine that can't hold 40 fps keeps the
 * scene but drops to dpr 1 without bloom.
 */

type Controls = { collapse: number; pointerX: number; pointerY: number };
type Vec3 = [number, number, number];

const vert = /* glsl */ `
  uniform float uTime;
  uniform float uSpeed;
  uniform float uPixel;
  uniform float uCollapse;
  uniform float uSpin;
  attribute vec3 aOuter;
  attribute float aSeed;
  attribute float aSpin;
  varying float vT;
  varying float vPull;

  vec3 rotY(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z); }

  void main() {
    float t = fract(uTime * uSpeed * (0.6 + 0.4 * aSpin) + aSeed);
    // slow drift in the cloud, then a decisive pull into the core
    float pull = smoothstep(0.35, 1.0, t);
    pull = pull * pull * (3.0 - 2.0 * pull);
    pull = mix(pull, 1.0, uCollapse);
    vec3 outer = rotY(aOuter, uTime * uSpin + aSeed * 6.2831);
    outer += 0.18 * vec3(sin(uTime * 0.7 + aSeed * 40.0), cos(uTime * 0.5 + aSeed * 31.0), sin(uTime * 0.6 + aSeed * 17.0));
    vec3 core = normalize(aOuter) * (0.46 + 0.16 * aSpin);
    vec3 p = mix(outer, core, pull);
    p = rotY(p, pull * (2.4 + aSpin * 2.0));
    vT = t;
    vPull = pull;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float s = mix(1.7, 2.8, pull) * (0.6 + aSpin * 0.8);
    gl_PointSize = s * uPixel * (6.5 / -mv.z);
  }
`;

const frag = /* glsl */ `
  uniform vec3 uCloud;
  uniform vec3 uCore;
  uniform float uCollapse;
  varying float vT;
  varying float vPull;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float hot = smoothstep(0.55, 1.0, vPull);
    vec3 col = mix(uCloud, uCore, hot);
    float fadeIn = smoothstep(0.0, 0.08, vT);
    float fadeOut = 1.0 - smoothstep(0.96, 1.0, vT);
    float cycle = mix(fadeIn * fadeOut, 1.0, uCollapse);
    float alpha = (1.0 - smoothstep(0.15, 0.5, d)) * mix(0.34, 0.95, hot) * cycle;
    gl_FragColor = vec4(col, alpha);
  }
`;

function buildGeometry(count: number) {
  const g = new THREE.BufferGeometry();
  const outer = new Float32Array(count * 3);
  const seed = new Float32Array(count);
  const spin = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    // flattened, lumpy shell: wide like a general model's knowledge, not a perfect sphere
    const u = Math.random() * 2 - 1;
    const th = Math.random() * Math.PI * 2;
    const r = 2.4 + Math.pow(Math.random(), 0.6) * 2.6;
    const s = Math.sqrt(1 - u * u);
    outer[i * 3] = r * s * Math.cos(th) * 1.35;
    outer[i * 3 + 1] = r * u * 0.62;
    outer[i * 3 + 2] = r * s * Math.sin(th);
    seed[i] = Math.random();
    spin[i] = Math.random();
  }
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  g.setAttribute("aOuter", new THREE.BufferAttribute(outer, 3));
  g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
  g.setAttribute("aSpin", new THREE.BufferAttribute(spin, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 8);
  return g;
}

function Particles({ count, reduced, controls }: { count: number; reduced: boolean; controls: RefObject<Controls> }) {
  const mat = useRef<THREE.ShaderMaterial>(null);
  const gl = useThree((s) => s.gl);
  const dpr = useThree((s) => s.viewport.dpr);
  const geometry = useMemo(() => buildGeometry(count), [count]);
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uSpeed: { value: reduced ? 0.012 : 0.05 },
      uSpin: { value: reduced ? 0 : 0.05 },
      uPixel: { value: Math.min(gl.getPixelRatio(), 2) },
      uCollapse: { value: 0 },
      uCloud: { value: new THREE.Color("#8d8782") },
      uCore: { value: new THREE.Color("#e4f222").multiplyScalar(1.35) },
    }),
    [gl, reduced],
  );

  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame((_, delta) => {
    const u = mat.current?.uniforms;
    if (!u) return;
    const d = Math.min(delta, 0.05);
    u.uPixel.value = Math.min(dpr, 2);  // point size is in device pixels; follows the monitor when it steps dpr down
    u.uTime.value += d;
    u.uCollapse.value += (controls.current.collapse - u.uCollapse.value) * (1 - Math.exp(-d * 6));
  });

  return (
    <points geometry={geometry}>
      <shaderMaterial ref={mat} vertexShader={vert} fragmentShader={frag} uniforms={uniforms}
        transparent depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  );
}

function Core({ reduced, controls }: { reduced: boolean; controls: RefObject<Controls> }) {
  const inner = useRef<THREE.Group>(null);
  const shell = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (!inner.current || !shell.current) return;
    const spin = reduced ? 0.04 : 1;
    inner.current.rotation.y += delta * 0.35 * spin;
    inner.current.rotation.x += delta * 0.12 * spin;
    shell.current.rotation.y -= delta * 0.08 * spin;
    shell.current.rotation.z += delta * 0.05 * spin;
    const k = 1 + controls.current.collapse * 0.45;
    inner.current.scale.setScalar(k);
  });
  return (
    <>
      <group ref={inner}>
        <mesh>
          <icosahedronGeometry args={[0.38, 1]} />
          <meshBasicMaterial color={[2.2, 2.35, 0.3]} wireframe transparent opacity={0.9} toneMapped={false} />
        </mesh>
        <mesh>
          <icosahedronGeometry args={[0.2, 0]} />
          <meshBasicMaterial color={[3.2, 3.4, 1.1]} toneMapped={false} />
        </mesh>
      </group>
      {/* the structure that holds the core: the logo's outer shape, in three dimensions */}
      <mesh ref={shell} rotation={[0.4, 0.2, 0]}>
        <icosahedronGeometry args={[1.05, 1]} />
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.1} />
      </mesh>
    </>
  );
}

function Rig({ children, controls, offset, offsetMobile, interactive }: {
  children: React.ReactNode; controls: RefObject<Controls>; offset: Vec3; offsetMobile: Vec3; interactive: boolean;
}) {
  const ref = useRef<THREE.Group>(null);
  const width = useThree((s) => s.size.width);
  const pos = width < 700 ? offsetMobile : offset;
  useEffect(() => {
    if (!interactive) return;
    const on = (e: PointerEvent) => {
      controls.current.pointerX = (e.clientX / window.innerWidth - 0.5) * 0.5;
      controls.current.pointerY = (e.clientY / window.innerHeight - 0.5) * 0.3;
    };
    window.addEventListener("pointermove", on, { passive: true });
    return () => window.removeEventListener("pointermove", on);
  }, [interactive, controls]);
  useFrame((_, d) => {
    if (!ref.current) return;
    const k = 1 - Math.exp(-d * 2.5);
    const c = controls.current;
    ref.current.rotation.y += (c.pointerX - ref.current.rotation.y) * k;
    ref.current.rotation.x += (c.pointerY + 0.18 - ref.current.rotation.x) * k;
  });
  return <group ref={ref} position={pos}>{children}</group>;
}

// The probe context is released at once so it never competes with the real canvas for a GPU slot.
function hasWebGL(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const probe = document.createElement("canvas");
    const gl = probe.getContext("webgl2") ?? probe.getContext("webgl");
    if (!gl) return false;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

// Renderer construction can still throw after a passing probe (blocked GPU, exhausted contexts); that lands here.
class SceneBoundary extends Component<{ onError: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function Fallback() {
  const spot = "absolute left-1/2 top-[70%] -translate-x-1/2 -translate-y-1/2 md:left-[68%] md:top-1/2";
  return (
    <div data-testid="scene-fallback" className="absolute inset-0 overflow-hidden bg-ink">
      <div className={`${spot} h-[80vmin] w-[80vmin] rounded-full bg-[radial-gradient(closest-side,rgba(228,242,34,0.14),rgba(228,242,34,0.03)_55%,transparent_100%)]`} />
      <Mark className={`${spot} h-[18vmin] w-[18vmin] text-white/35`} strokeWidth={1.1} />
    </div>
  );
}

export default function DistillScene({
  density = 1, className, offset = [0, 0, 0], offsetMobile = offset, collapseOnScroll = false, interactive = true, bloom = true,
}: {
  density?: number; className?: string; offset?: Vec3; offsetMobile?: Vec3; collapseOnScroll?: boolean; interactive?: boolean; bloom?: boolean;
}) {
  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const count = Math.round((typeof window !== "undefined" && window.innerWidth < 700 ? 5000 : 14000) * density);
  const controls = useRef<Controls>({ collapse: 0, pointerX: 0, pointerY: 0 });
  const host = useRef<HTMLDivElement>(null);
  const [onScreen, setOnScreen] = useState(true);
  const [webgl, setWebgl] = useState(hasWebGL);
  const [degraded, setDegraded] = useState(false);

  // Render only while visible: a hero that scrolled away shouldn't keep the GPU busy.
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setOnScreen(e.isIntersecting), { threshold: 0.02 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!collapseOnScroll) return;
    const on = () => {
      controls.current.collapse = Math.min(1, Math.max(0, window.scrollY / (window.innerHeight * 0.85)));
    };
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, [collapseOnScroll]);

  return (
    <div ref={host} className={className} aria-hidden>
      {webgl ? (
        <SceneBoundary onError={() => setWebgl(false)}>
          <Canvas
            frameloop={onScreen ? "always" : "never"}
            dpr={degraded ? 1 : [1, 1.5]}
            camera={{ position: [0, 0.15, 7.6], fov: 40 }}
            gl={{ antialias: false, alpha: false, powerPreference: "high-performance" }}
            onCreated={({ gl }) => gl.domElement.addEventListener("webglcontextlost", () => setWebgl(false), { once: true })}
          >
            <color attach="background" args={["#0c0a08"]} />
            <PerformanceMonitor onDecline={() => setDegraded(true)} />
            <Suspense fallback={null}>
              <Rig controls={controls} offset={offset} offsetMobile={offsetMobile} interactive={interactive && !reduced}>
                <Particles count={count} reduced={reduced} controls={controls} />
                <Core reduced={reduced} controls={controls} />
              </Rig>
              {bloom && !degraded && (
                <EffectComposer multisampling={0}>
                  <Bloom luminanceThreshold={1} luminanceSmoothing={0.25} intensity={0.9} mipmapBlur radius={0.6} />
                </EffectComposer>
              )}
            </Suspense>
          </Canvas>
        </SceneBoundary>
      ) : (
        <Fallback />
      )}
    </div>
  );
}
