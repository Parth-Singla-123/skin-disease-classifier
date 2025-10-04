"use client";

import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(""); // Clear previous result
    }
  };


  const handleDetect = async () => {
    if (!selectedFile) {
      alert("Please upload an image first!");
      return;
    }
  
    setLoading(true);
    setResult("🔍 Detecting disease...");
  
    try {
      // Prepare FormData
      const formData = new FormData();
      formData.append("file", selectedFile);
  
      // Call Flask API
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
  
      const data = await response.json();
      console.log("API Response:", data);
  
      // Update result
      setResult(`✅ Detected: ${data.disease} (Confidence: ${(data.confidence * 100).toFixed(2)}%)`);
    } catch (error) {
      console.error("Error during API call:", error);
      setResult("❌ Failed to detect disease. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-black to-gray-900 text-white px-4">
      <div className="w-full max-w-xl rounded-3xl bg-gradient-to-br from-gray-800 to-gray-900 shadow-2xl p-8 border border-gray-700">
        <h1 className="text-4xl font-extrabold text-center text-cyan-400 mb-4 tracking-wide animate-pulse">
          Skin Disease Detection
        </h1>
        <p className="text-center text-gray-300 mb-6">
          Upload an image of the affected skin area to get an AI diagnosis.
        </p>

        <div className="flex flex-col gap-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="border border-gray-600 rounded-lg p-2 bg-gray-800 text-gray-200 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-cyan-600 file:text-white hover:file:bg-cyan-700 transition-all duration-300"
          />

          {previewUrl && (
            <div className="relative w-full h-64 rounded-xl overflow-hidden border border-gray-700">
              <Image
                src={previewUrl}
                alt="Preview"
                fill
                className="object-cover"
              />
            </div>
          )}

          <button
            onClick={handleDetect}
            className={`relative bg-cyan-600 hover:bg-cyan-700 text-white py-3 px-6 rounded-xl font-semibold text-lg transition-transform transform hover:scale-105 active:scale-95 shadow-lg shadow-cyan-500/20 ${
              loading ? "opacity-70 cursor-not-allowed" : ""
            }`}
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg
                  className="w-5 h-5 animate-spin text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v8H4z"
                  ></path>
                </svg>
                Detecting...
              </span>
            ) : (
              "Detect Disease"
            )}
          </button>

          {result && (
            <div
              className={`mt-4 p-4 rounded-xl border ${
                result.includes("✅")
                  ? "bg-green-800/20 text-green-400 border-green-600"
                  : "bg-yellow-800/20 text-yellow-400 border-yellow-600"
              } animate-fade-in`}
            >
              {result}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
