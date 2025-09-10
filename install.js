module.exports = {
  run: [
    
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true
          // triton: true
          // sageattention: true
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python - <<'PY'\nimport whisper; whisper.load_model('large-v3')\nPY",
          "python - <<'PY'\nfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('ai4bharat/indictrans2-indic-indic-1B', trust_remote_code=True); AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-indic-indic-1B', trust_remote_code=True)\nPY",
          "python - <<'PY'\nfrom transformers import AutoModel; AutoModel.from_pretrained('6Morpheus6/IndicF5', trust_remote_code=True)\nPY",
          "uv pip install -r requirements.txt --no-cache"
        ],
      }
    }
  ]
}
