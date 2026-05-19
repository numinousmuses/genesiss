import Editor from "@monaco-editor/react";

interface Props {
  value: string;
  onChange: (v: string) => void;
}

export function CodeView({ value, onChange }: Props) {
  return (
    <Editor
      height="100%"
      defaultLanguage="python"
      value={value}
      theme="vs-dark"
      onChange={(v) => onChange(v ?? "")}
      options={{
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
        wordWrap: "on",
        automaticLayout: true,
      }}
    />
  );
}
