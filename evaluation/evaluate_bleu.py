from nltk.translate.bleu_score import sentence_bleu
from src.response_generator import generate_response

reference = ["You can return the product within 30 days".split()]
candidate = generate_response("How do I return a product?").split()
print("BLEU Score:", sentence_bleu(reference, candidate))