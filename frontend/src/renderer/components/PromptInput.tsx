import { useState } from "react";

interface Props {
  disabled: boolean;
  onSubmit: (prompt: string) => void;
}

export function PromptInput({ disabled, onSubmit }: Props) {
  const [value, setValue] = useState("");

  function go() {
    const t = value.trim();
    if (!t) return;
    onSubmit(t);
    setValue("");
  }

  return (
    <div className="prompt-row">
      <input
        type="text"
        placeholder="e.g. a 40mm hex nut with M10 threading"
        value={value}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            go();
          }
        }}
      />
      <button disabled={disabled || !value.trim()} onClick={go}>
        Generate
      </button>
    </div>
  );
}
