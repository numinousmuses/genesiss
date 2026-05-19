import { Canvas } from "@react-three/fiber";
import { Bounds, Grid, OrbitControls } from "@react-three/drei";
import { useMemo } from "react";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { BufferGeometry } from "three";

interface Props {
  stlB64: string | null;
}

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function Mesh({ geom }: { geom: BufferGeometry }) {
  return (
    <mesh geometry={geom} castShadow receiveShadow>
      <meshStandardMaterial color="#cfe9d8" metalness={0.1} roughness={0.6} />
    </mesh>
  );
}

export function ModelViewer({ stlB64 }: Props) {
  const geom = useMemo(() => {
    if (!stlB64) return null;
    const loader = new STLLoader();
    const buf = b64ToBytes(stlB64).buffer;
    const g = loader.parse(buf);
    g.computeVertexNormals();
    g.center();
    return g;
  }, [stlB64]);

  return (
    <Canvas shadows camera={{ position: [80, 80, 80], fov: 35 }}>
      <ambientLight intensity={0.4} />
      <directionalLight position={[60, 80, 40]} intensity={0.9} castShadow />
      <Grid
        args={[400, 400]}
        cellThickness={0.6}
        sectionThickness={1.2}
        sectionColor="#3a3a48"
        cellColor="#1f1f28"
        fadeDistance={300}
        infiniteGrid
      />
      <Bounds fit clip margin={1.2}>
        {geom ? <Mesh geom={geom} /> : null}
      </Bounds>
      <OrbitControls makeDefault enableDamping />
    </Canvas>
  );
}
