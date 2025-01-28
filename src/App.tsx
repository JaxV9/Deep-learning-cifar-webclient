import './App.css'
import { useEffect, useState } from "react";
// import * as ort from 'onnxruntime-web/wasm';
import * as ort from 'onnxruntime-web';
function App() {

  const [output, setOutput] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const cifar_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
  ]

  const preprocessImage = (image: HTMLImageElement): ort.Tensor => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = 32;
    canvas.height = 32;
    ctx?.drawImage(image, 0, 0, 32, 32);

    const imageData = ctx?.getImageData(0, 0, 32, 32);
    if (!imageData) throw new Error("ImageData is null");

    const data = new Float32Array(3 * 32 * 32);

    for (let i = 0; i < 32 * 32; i++) {
      const pixelIndex = i * 4;
      data[i] = imageData.data[pixelIndex] / 255.0;
      data[i + 32 * 32] = imageData.data[pixelIndex + 1] / 255.0;
      data[i + 2 * 32 * 32] = imageData.data[pixelIndex + 2] / 255.0;
    }

    return new ort.Tensor("float32", data, [1, 3, 32, 32]);
  };

  const runModel = async (tensor: ort.Tensor) => {
    setLoading(true);
    try {

      const session = await ort.InferenceSession.create('/model.onnx');

      // Effectuer l'inférence avec le modèle ONNX
      const feeds = { input: tensor };  // Assurez-vous que 'input' est le bon nom de l'entrée du modèle
      const results = await session.run(feeds);

      const outputData = results.output;  // récupère les résultats de l'inférence
      // Traitement des résultats
      const maxIndex = (outputData.data as Float32Array).indexOf(Math.max(...(outputData.data as Float32Array)));
      setOutput(cifar_classes[maxIndex]);
    } catch (err) {
      console.error("Erreur lors de l'inférence :", err);
    } finally {
      setLoading(false);
    }
  };

  const loadImage = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const image = new Image();
      image.src = URL.createObjectURL(file);
      image.onload = async () => {
        const tensor = preprocessImage(image);
        await runModel(tensor);
      };
    }
  };

  useEffect(() => {
    const loadModel = async () => {
      try {
        // Chemin relatif vers votre modèle ONNX
        ort.env.wasm.wasmPaths = '/';
        ort.env.wasm.simd = false;
        ort.env.wasm.numThreads = 1;
      } catch {
        console.log("error")
      }
    };
    loadModel();
  }, []);

  return (
    <>
      <h1>CIFAR10 AI</h1>
      <h3>By LAYAN Jason</h3>
      <h3>Output:</h3>
      <p>Réponses possibles :</p>
      <p>
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"</p>
      <input type="file" accept="image/*" onChange={loadImage} />
/* {loading && <p>Chargement...</p>}
<pre>{JSON.stringify(output, null, 2)}</pre>

</>
)
}

export default App
