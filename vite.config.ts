import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),
  ],
  server: {
    fs: {
      // Permet de servir des fichiers depuis le dossier public
      strict: false,
    },
    // mimeTypes: {
    //   '.wasm': 'application/wasm', // Définir le type MIME pour .wasm
    // },
  },
})
