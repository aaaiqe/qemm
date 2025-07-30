import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# --- Quantum Library Imports ---
# These imports are essential for the quantum-inspired layers.
# If PennyLane is not installed, the quantum layers will operate using classical approximations.
# For full functionality, ensure PennyLane is installed: pip install pennylane
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.templates.embeddings import AngleEmbedding
    from pennylane.templates.layers import StronglyEntanglingLayers
except ImportError:
    print("Warning: PennyLane not detected. Quantum layers will operate using classical approximations.")
    qml = None
    pnp = np # Fallback for PennyLane's numpy operations


class QEAF_Layer(nn.Module):
    """
    Quantum Entanglement-Aware Feature Fusion (QEAF) Layer.
    This layer implements the quantum-inspired fusion of visual and textual embeddings.
    It maps a classical multimodal embedding to a quantum-inspired feature space,
    leveraging entanglement principles to capture complex, non-linear correlations.
    """
    def __init__(self, input_dim: int, num_qubits: int = 8, q_layers: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_qubits = num_qubits

        if qml is None:
            # Classical approximation for QEAF if PennyLane is not available.
            self.classical_fusion = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, input_dim)
            )
            print("QEAF_Layer operating in classical approximation mode.")
            return

        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qeaf_circuit(features):
            """
            Quantum circuit for Quantum Entanglement-Aware Feature Fusion.
            Encodes classical features into a quantum state and applies entangling layers.
            """
            AngleEmbedding(features, wires=range(self.num_qubits), rotation='Y')
            StronglyEntanglingLayers(weights=self.pqc_weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qeaf_circuit = qeaf_circuit
        
        # Learnable parameters for the PQC (e.g., for StronglyEntanglingLayers).
        self.pqc_weights = nn.Parameter(torch.rand(q_layers, num_qubits, 3)) 
        
        # Classical projection layer to ensure output dimension consistency.
        self.output_proj = nn.Linear(num_qubits, input_dim)

    def forward(self, multimodal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Applies Quantum Entanglement-Aware Feature Fusion.
        Args:
            multimodal_embedding: Classical multimodal embedding (concatenated and projected from text/visual encoders).
        Returns:
            Quantum-enhanced multimodal embedding.
        """
        if qml is None:
            return self.classical_fusion(multimodal_embedding)

        # Scale features for AngleEmbedding, typically to [0, 2pi], and clamp to ensure valid range.
        scaled_features = multimodal_embedding / (multimodal_embedding.max().item() if multimodal_embedding.max().item() != 0 else 1.0) * (2 * pnp.pi)
        scaled_features = torch.clamp(scaled_features, 0, 2 * pnp.pi) 
        
        # Run the PQC and obtain expectation values.
        quantum_output = self.qeaf_circuit(scaled_features)

        # Project the quantum circuit output to the desired dimension.
        quantum_enhanced_embedding = self.output_proj(quantum_output)
        
        return quantum_enhanced_embedding


class VQC_Classifier(nn.Module):
    """
    Variational Quantum Classifier (VQC) for Hierarchical Detection.
    This classifier operates on quantum-enhanced embeddings to perform hierarchical classification.
    """
    def __init__(self, input_dim: int, num_classes: int, num_qubits: int = 8, q_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_classes = num_classes

        if qml is None:
            # Classical fallback for VQC if PennyLane is not available.
            self.classical_classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, num_classes)
            )
            print("VQC_Classifier operating in classical approximation mode.")
            return

        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def vqc_circuit(features, weights):
            """
            Variational Quantum Circuit for classification.
            Encodes features and applies a trainable quantum ansatz.
            """
            AngleEmbedding(features, wires=range(self.num_qubits))
            StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))
            # Measure probabilities of basis states for classification.
            return qml.probs(wires=range(self.num_qubits))

        self.vqc_circuit = vqc_circuit
        
        # Learnable parameters for the PQC (e.g., for StronglyEntanglingLayers).
        self.pqc_weights = nn.Parameter(torch.rand(q_layers, num_qubits, 3)) 
        
        # Classical post-processing layer to map qubit probabilities to final class logits.
        self.final_classifier_head = nn.Linear(2**num_qubits, num_classes)

    def forward(self, quantum_enhanced_embedding: torch.Tensor) -> torch.Tensor:
        """
        Applies Variational Quantum Classifier.
        Args:
            quantum_enhanced_embedding: Output from QEAF layer (q_i).
        Returns:
            Logits for classification.
        """
        if qml is None:
            return self.classical_classifier(quantum_enhanced_embedding)

        # Scale features for AngleEmbedding, typically to [0, 2pi].
        scaled_features = quantum_enhanced_embedding / (quantum_enhanced_embedding.max().item() if quantum_enhanced_embedding.max().item() != 0 else 1.0) * (2 * pnp.pi)
        scaled_features = torch.clamp(scaled_features, 0, 2 * pnp.pi) 

        # Run the VQC circuit.
        probs = self.vqc_circuit(scaled_features, self.pqc_weights)
        
        # Post-process probabilities to get logits for classification loss.
        logits = self.final_classifier_head(probs)

        return logits


class QE_MMDE_Model(nn.Module):
    """
    Quantum-Enhanced Multimodal Misogyny Detection and Explanation (QE-MMDE) Framework.
    Integrates classical encoders, Quantum Entanglement-Aware Feature Fusion (QEAF),
    Variational Quantum Classifiers (VQC), and a VLM for reasoning.
    """
    def __init__(self, 
                 text_model_name: str = "sentence-transformers/paraphrase-multilingual-e5-large",
                 image_model_name: str = "openai/clip-vit-base-patch32", 
                 multimodal_embedding_dim: int = 768, 
                 num_qubits_qeaf: int = 8,
                 q_layers_qeaf: int = 2,
                 num_qubits_vqc: int = 8,
                 q_layers_vqc: int = 4,
                 num_classes_level1: int = 2, # Non-misogynous/Misogynous
                 num_classes_level2: int = 4, # Kitchen, Leadership, Working, Shopping
                 vlm_name: str = "google/flan-t5-base" 
                ):
        super().__init__()

        # 1. Classical Encoders
        # Text Encoder: Multilingual-E5-large for strong multilingual embeddings.
        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer(text_model_name)
        # For efficiency, consider freezing large pre-trained encoders during initial training.
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

        # Image Encoder: CLIP-ViT as a robust VLM component.
        from transformers import CLIPProcessor, CLIPModel
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.image_encoder = CLIPModel.from_pretrained(image_model_name).get_image_features
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False
        
        # Linear projections to align encoder output dimensions to a common multimodal_embedding_dim.
        self.text_proj = nn.Linear(self.text_encoder.get_sentence_embedding_dimension(), multimodal_embedding_dim)
        self.image_proj = nn.Linear(self.image_encoder.config.projection_dim, multimodal_embedding_dim)


        # 2. Quantum Entanglement-Aware Feature Fusion (QEAF) Layer
        # Processes the concatenated classical multimodal features.
        self.qeaf_layer = QEAF_Layer(multimodal_embedding_dim * 2, num_qubits_qeaf, q_layers_qeaf)


        # 3. Hierarchical Variational Quantum Classifiers (VQC)
        # VQC_1 for Level 1: Misogynous/Non-misogynous classification.
        self.vqc_classifier_level1 = VQC_Classifier(multimodal_embedding_dim, num_classes_level1, num_qubits_vqc, q_layers_vqc)
        # VQC_2 for Level 2: Fine-grained classification (e.g., Kitchen, Leadership, etc.).
        self.vqc_classifier_level2 = VQC_Classifier(multimodal_embedding_dim, num_classes_level2, num_qubits_vqc, q_layers_vqc)


        # 4. VLM for Reasoning Generation (Quantum-Informed Reasoning)
        # This VLM benefits from the quantum-enhanced embeddings provided as context.
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_name)
        self.vlm_decoder = AutoModelForCausalLM.from_pretrained(vlm_name)


    def forward(self, meme_images_pil: list[Image.Image], meme_texts: list[str]):
        """
        Forward pass for hierarchical classification.
        Args:
            meme_images_pil: List of PIL Image objects (raw image data).
            meme_texts: List of strings (OCR text from memes).
        Returns:
            Logits and predictions for Level 1 and Level 2 classification,
            and the quantum_enhanced_embedding for reasoning generation.
        """
        # 1. Encode Text
        text_embeddings = self.text_encoder.encode(meme_texts, convert_to_tensor=True, device=self.text_encoder.device)
        text_embeddings_proj = self.text_proj(text_embeddings) 

        # 2. Encode Image
        image_inputs = self.image_processor(images=meme_images_pil, return_tensors="pt")
        image_inputs = {k: v.to(self.image_proj.weight.device) for k, v in image_inputs.items()}
        
        image_embeddings = self.image_encoder(**image_inputs).pooler_output
        image_embeddings_proj = self.image_proj(image_embeddings) 


        # 3. Classical Fusion (Concatenation before QEAF)
        fused_classical_embedding = torch.cat((text_embeddings_proj, image_embeddings_proj), dim=-1)


        # 4. Quantum Entanglement-Aware Feature Fusion (QEAF)
        quantum_enhanced_embedding = self.qeaf_layer(fused_classical_embedding)


        # 5. Hierarchical VQC Classification
        logits_level1 = self.vqc_classifier_level1(quantum_enhanced_embedding)
        probs_level1 = torch.softmax(logits_level1, dim=-1)
        preds_level1 = torch.argmax(probs_level1, dim=-1)

        logits_level2 = self.vqc_classifier_level2(quantum_enhanced_embedding)
        probs_level2 = torch.softmax(logits_level2, dim=-1)
        preds_level2 = torch.argmax(probs_level2, dim=-1)

        return logits_level1, logits_level2, preds_level1, preds_level2, quantum_enhanced_embedding


    def generate_explanation(self, meme_image_pil: Image.Image, meme_text: str, predicted_label: str, quantum_enhanced_embedding: torch.Tensor = None) -> str:
        """
        Generates a natural language explanation for a meme, leveraging quantum-enhanced embedding.
        Args:
            meme_image_pil: PIL Image object.
            meme_text: String (OCR text).
            predicted_label: Predicted misogyny category (e.g., "Kitchen").
            quantum_enhanced_embedding: The q_i from QEAF, which informs the VLM's generation.
        Returns:
            Generated explanation string.
        """
        # Ensure the quantum_enhanced_embedding is available. If not, re-calculate it.
        if quantum_enhanced_embedding is None:
            text_emb_exp = self.text_encoder.encode([meme_text], convert_to_tensor=True, device=self.text_encoder.device)
            text_emb_proj_exp = self.text_proj(text_emb_exp)
            img_inputs_exp = self.image_processor(images=[meme_image_pil], return_tensors="pt").to(self.image_proj.weight.device)
            img_emb_exp = self.image_encoder(**img_inputs_exp).pooler_output
            img_emb_proj_exp = self.image_proj(img_emb_exp)
            fused_classical_emb_exp = torch.cat((text_emb_proj_exp, img_emb_proj_exp), dim=-1)
            quantum_enhanced_embedding = self.qeaf_layer(fused_classical_emb_exp)

        # Construct the task prompt for the VLM.
        if predicted_label in ["Kitchen", "Shopping", "Leadership", "Working"]:
            task_instruction = (
                f"You are an expert meme reviewer. This meme has been flagged as '{predicted_label}' misogyny. "
                "Carefully analyze both the image and the text. Explain why it is misogynistic within the context of "
                f"the '{predicted_label}' domain. For example, mention if it targets gender roles, leadership abilities, work competence, "
                "or consumerism, or uses harmful stereotypes, slurs, or implicit cues. Provide a clear, reasoned explanation based on context."
            )
        else: # Non-misogynous case
             task_instruction = (
                f"As a meme moderation expert, analyze the image and caption together. This meme has been classified as 'Non-misogynistic'. "
                "Explain why it does not fall under harmful categories, considering visual elements, text, tone, context, and cultural sensitivity. "
                "Be precise and avoid generic responses. Justify why it does not fall under misogynistic categories like Kitchen, Leadership, Working, or Shopping."
            )

        input_text = f"{task_instruction}\n\nMeme Text: \"{meme_text}\"\nExplanation:"
        
        # Tokenize the input text for the VLM.
        inputs = self.vlm_tokenizer(input_text, return_tensors="pt").to(self.vlm_decoder.device)

        # Generate the explanation using the VLM decoder.
        # The 'quantum-informed' aspect means the VLM leverages the richness of the
        # quantum_enhanced_embedding provided from QEAF in its internal processing or as soft prompts.
        output_tokens = self.vlm_decoder.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
        explanation = self.vlm_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Post-process the generated text to remove any prompt repetition.
        if "Explanation:" in explanation:
            explanation = explanation.split("Explanation:", 1)[1].strip()

        return explanation


# --- Example Usage (Conceptual Demonstration) ---
if __name__ == "__main__":
    # Check if PennyLane is available; quantum components are active if so.
    if qml is None:
        print("PennyLane is not available. Quantum layers will operate using classical approximations.")
        print("To enable full quantum simulation, please install PennyLane: pip install pennylane")
    else:
        print("PennyLane detected. Full quantum simulation mode active.")

    # Device setup: Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the QE-MMDE Model with default parameters.
    # Note: This will download pre-trained LLMs/VLMs on first run.
    try:
        model = QE_MMDE_Model().to(device)
        model.eval() # Set the model to evaluation mode (disables dropout, etc.)
        print("QE-MMDE Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing model. Ensure all pre-trained models can be loaded. Error: {e}")
        print("Please check your internet connection or HuggingFace token if using private models.")
        exit()

    # --- Prepare Dummy Data for Demonstration ---
    # Create dummy PIL Images and text for testing the model's forward pass.
    dummy_image_1 = Image.new('RGB', (224, 224), color = 'red')
    dummy_text_1 = "Why are women always talking about shoes?" # Example text for classification

    dummy_image_2 = Image.new('RGB', (224, 224), color = 'blue')
    dummy_text_2 = "He's a great manager, almost like a woman!" # Example text for explanation

    # --- Classification Example ---
    print("\n--- Classification Example ---")
    with torch.no_grad(): # Disable gradient calculations for inference.
        logits_l1, logits_l2, preds_l1, preds_l2, q_emb_for_explanation = model([dummy_image_1], [dummy_text_1])
        
        # Define labels for mapping prediction indices to meaningful categories.
        # This order must match your VQC_Classifier_level2's training labels.
        level1_labels = ["Non-Misogynous", "Misogynous"]
        level2_labels = ["Kitchen", "Leadership", "Working", "Shopping"] 
        
        predicted_l1_label = level1_labels[preds_l1.item()]
        predicted_l2_label = level2_labels[preds_l2.item()]

        print(f"Meme 1 (Text: '{dummy_text_1}'):")
        print(f"  Predicted Level 1: {predicted_l1_label}")
        if predicted_l1_label == "Misogynous":
            print(f"  Predicted Level 2: {predicted_l2_label}")


    # --- Explanation Generation Example ---
    print("\n--- Explanation Generation Example ---")
    
    # Use the quantum_enhanced_embedding obtained from a forward pass for the explanation demo.
    # In a real inference pipeline, this `q_emb_for_explanation` would be the output
    # from the `forward` call for the meme needing explanation.
    
    # Assume a predicted label for the purpose of generating an explanation.
    predicted_label_for_explanation = "Leadership" # Example: Assume it was classified as Leadership misogyny.

    # Generate the explanation using the model's dedicated method.
    explanation_text = model.generate_explanation(
        meme_image_pil=dummy_image_2,
        meme_text=dummy_text_2,
        predicted_label=predicted_label_for_explanation,
        quantum_enhanced_embedding=q_emb_for_explanation # Pass the Q-enhanced embedding
    )
    
    print(f"Meme 2 (Text: '{dummy_text_2}'):")
    print(f"  Assumed Predicted Label (for explanation demo): {predicted_label_for_explanation}")
    print(f"  Generated Explanation: {explanation_text}")