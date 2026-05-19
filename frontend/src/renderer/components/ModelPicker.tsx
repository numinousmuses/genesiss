import type { Variant } from "../lib/api";

interface Props {
  variants: Variant[];
  value: string;
  onChange: (v: string) => void;
}

export function ModelPicker({ variants, value, onChange }: Props) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {variants.map((v) => (
        <option key={v.name} value={v.name}>
          {v.name}
        </option>
      ))}
    </select>
  );
}
