# embed_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/embed", methods=["POST"])
def embed():

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "no json body"}), 400

    if "texts" in data and isinstance(data["texts"], list):
        texts = data["texts"]
        if len(texts) == 0:
            return jsonify({"error": "empty texts list"}), 400
        embs = model.encode(texts, convert_to_numpy=True)
        # convert to list-of-lists
        out = [emb.tolist() for emb in embs]
        return jsonify({"embeddings": out}), 200

    text = data.get("text", None)
    if text is None:
        return jsonify({"error": "no 'text' or 'texts' field provided"}), 400
    embedding = model.encode(text, convert_to_numpy=True).tolist()
    return jsonify({"embedding": embedding}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=6000, debug=True)
