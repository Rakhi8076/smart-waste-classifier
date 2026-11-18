import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Prediction failed");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Could not reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Smart Waste Classifier</h1>
      <p className="subtitle">Upload a photo of waste to see its category and how to dispose of it.</p>

      <input type="file" accept="image/*" onChange={handleFile} />

      {preview && <img className="preview" src={preview} alt="Selected waste" />}

      <button onClick={handleSubmit} disabled={!file || loading}>
        {loading ? "Analyzing..." : "Classify"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <h2 className="category">{result.category}</h2>
          <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          <p>
            <span className={result.biodegradable ? "badge green" : "badge red"}>
              {result.biodegradable ? "Biodegradable" : "Non-biodegradable"}
            </span>
          </p>

          {result.message && <p className="warn">{result.message}</p>}

          <h3>Where to dispose</h3>
          <p>{result.disposal.bin}</p>
          <ul>
            {result.disposal.tips.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
          <p className="note">Rules can differ by city, so check your local municipal guidelines.</p>
        </div>
      )}
    </div>
  );
}