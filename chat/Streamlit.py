import streamlit as st
import pandas as pd
from typing import List, Dict, Set
import random
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

class ChatAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Please install the spacy model using: python -m spacy download en_core_web_sm")
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer()
        
        self.custom_stop_words = {
            'the', 'what', 'is', 'a', 'there', 'has', 'here', 'this', 'that',
            'these', 'those', 'am', 'be', 'been', 'being', 'was', 'were',
            'will', 'would', 'should', 'can', 'could', 'may', 'might',
            'must', 'shall', 'to', 'of', 'in', 'for', 'on', 'with'
        }
        
        for word in self.custom_stop_words:
            self.nlp.vocab[word].is_stop = True
        
        self.complexity_weights = {
            'basic': 0.5,
            'intermediate': 0.3,
            'advanced': 0.2
        }
        
        self.question_templates = {
            'basic': [
                "What is the definition of {concept}?",
                "Which of the following best describes {concept}?",
                "What role does {concept} play in {context}?",
                "What is the function of {concept} in {context}?",
                "How would you explain {concept}?"
            ],
            'intermediate': [
                "How does {concept} differ from {related_concept}?",
                "What are the benefits of using {concept} in {context}?",
                "In what situations would you use {concept} over {related_concept}?",
                "What are the key features of {concept} in {context}?",
                "How is {concept} applied in {context}?"
            ],
            'advanced': [
                "What are the limitations of {concept} in {context}?",
                "How does {concept} address issues in {related_concept}?",
                "What is the impact of {concept} in solving problems in {context}?",
                "What challenges exist when applying {concept} in {context}?",
                "How does {concept} compare to {related_concept} in {context}?"
            ]
        }
        
        self.concept_definitions = defaultdict(list)

    def preprocess(self, text: str) -> str:
        """
        Preprocess text by removing stop words and punctuation, and lemmatizing tokens.
        """
        filtered = []
        for token in self.nlp(text):
            if token.is_stop or token.is_punct:
                continue
            filtered.append(token.lemma_)
        return " ".join(filtered)

    def _extract_semantic_info(self, messages: List[str]) -> List[Dict]:
        """
        Extracts key terms from chat messages using preprocessing and TF-IDF.
        """
        # Preprocess all messages
        filtered_messages = [self.preprocess(msg) for msg in messages]
        
        # Apply TF-IDF vectorization
        vectorized_messages = self.tfidf.fit_transform(filtered_messages)
        
        # Get feature names and their scores
        feature_names = self.tfidf.get_feature_names_out()
        tfidf_scores = vectorized_messages.toarray()
        
        semantic_info = []
        for idx, (original_msg, filtered_msg) in enumerate(zip(messages, filtered_messages)):
            doc = self.nlp(original_msg)
            
            # Extract concepts with their TF-IDF scores
            concepts = []
            for chunk in doc.noun_chunks:
                if self._is_valid_concept(chunk):
                    cleaned_term = self.preprocess(chunk.text)
                    if cleaned_term:
                        # Get TF-IDF score for the term if it exists in features
                        importance = 0.0
                        for word in cleaned_term.split():
                            if word in feature_names:
                                word_idx = list(feature_names).index(word)
                                importance += tfidf_scores[idx][word_idx]
                        
                        if importance > 0:  # Only add terms with non-zero TF-IDF scores
                            concepts.append({
                                "term": cleaned_term,
                                "context": self._get_context(doc, chunk.root),
                                "sentence": original_msg,
                                "importance": importance
                            })
            
            # Sort concepts by TF-IDF importance
            concepts.sort(key=lambda x: x['importance'], reverse=True)
            
            info = {'text': original_msg, 'concepts': concepts}
            semantic_info.append(info)
            self._build_concept_definitions(info)
        
        return semantic_info

    def _is_valid_concept(self, chunk: spacy.tokens.Span) -> bool:
        """Check if a noun chunk is a valid concept."""
        preprocessed = self.preprocess(chunk.text)
        return bool(preprocessed)

    def _get_context(self, doc, token, window=5):
        """Extract context using preprocessing."""
        start = max(token.i - window, 0)
        end = min(token.i + window + 1, len(doc))
        context_text = doc[start:end].text
        return self.preprocess(context_text)

    def _build_concept_definitions(self, info):
        """Builds concept definitions for later use."""
        for concept in info['concepts']:
            term = concept['term']
            if term:  # Only add if term is not empty after preprocessing
                self.concept_definitions[term].append({
                    'definition': concept['context'],
                    'sentence': concept['sentence']
                })

    def _generate_mcq(self, info: Dict, complexity: str = 'basic') -> Dict:
        """Generates an MCQ using preprocessed concepts."""
        concept_info = random.choice(info['concepts'])
        concept, context = concept_info['term'], concept_info['context']
        related_concepts = [key for key in self.concept_definitions.keys() if key != concept]
        related_concept = random.choice(related_concepts) if related_concepts else "another concept"
        
        question_text = random.choice(self.question_templates[complexity]).format(
            concept=concept, 
            context=context, 
            related_concept=related_concept
        )
        correct_answer = f"{concept}: {self._get_definition_from_context(concept)}"
        incorrect_options = self._generate_incorrect_options(correct_answer, concept, related_concept)
        
        options = [correct_answer] + incorrect_options
        random.shuffle(options)
        
        return {
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "type": "mcq",
            "complexity": complexity,
            "concept": concept
        }

    def _generate_incorrect_options(self, correct_answer: str, concept: str, related_concept: str, num_options: int = 3) -> List[str]:
        incorrect_phrases = [
            lambda c, rc: f"is the same as {rc}",
            lambda c, rc: f"has no relation to {rc}",
            lambda c, rc: f"only applies to {rc}"
        ]
        incorrect_options = set()
        while len(incorrect_options) < num_options:
            wrong_def = random.choice(incorrect_phrases)(concept, related_concept)
            option = f"{concept} {wrong_def}"
            if option != correct_answer and option not in incorrect_options:
                incorrect_options.add(option)
        return list(incorrect_options)

    def _get_definition_from_context(self, concept: str) -> str:
        definitions = self.concept_definitions.get(concept, [])
        if definitions:
            return max(definitions, key=lambda x: len(x['definition']))['definition'].strip() + '.'
        return "Not well defined in the chat."

    def generate_mcqs(self, messages: List[str], num_questions: int = 5) -> List[Dict]:
        semantic_info = self._extract_semantic_info(messages)
        questions = []
        used_concepts = set()

        for _ in range(num_questions):
            info = random.choice(semantic_info)
            available_concepts = [c for c in info['concepts'] if c['term'] not in used_concepts]
            if not available_concepts:
                used_concepts.clear()
                available_concepts = info['concepts']
            
            complexity = random.choices(
                list(self.complexity_weights.keys()),
                weights=list(self.complexity_weights.values())
            )[0]
            question = self._generate_mcq({"concepts": available_concepts}, complexity)
            if question:
                used_concepts.add(question['concept'])
                questions.append(question)

        return questions
def displayAssignment(questions):
    """Display MCQ assignment with improved UI and feedback."""
    answers = {}
    st.header("MCQ Quiz")
    
    # Display the stats in a more appealing format
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", st.session_state["NumQuestions"])
    with col2:
        st.metric("Attempted", st.session_state["Attempted"])
    with col3:
        st.metric("Correct", st.session_state["Correct"])
    with col4:
        st.metric("Wrong", st.session_state["Wrong"])
    
    st.divider()
    
    # Display the questions with improved formatting
    for index, question in enumerate(questions):
        with st.container():
            st.subheader(f"Question {index + 1}")
            st.markdown(f"**{question['question']}**")
            st.markdown(f"*Complexity: {question['complexity'].title()}*")
            
            # Format options with proper spacing and numbering
            formatted_options = [f"{opt}" for opt in question["options"]]
            
            r = st.radio(
                "Select your answer:",
                formatted_options,
                index=None,
                key=f"q{index}"
            )
            
            if f"answered_q{index}" not in st.session_state:
                st.session_state[f"answered_q{index}"] = False

            if r and not st.session_state[f"answered_q{index}"]:
                st.session_state[f"answered_q{index}"] = True
                st.session_state["Attempted"] += 1
                
                if r == question["correct_answer"]:
                    st.success("✔️ Correct!", icon="✅")
                    st.session_state["Correct"] += 1
                else:
                    st.error("❌ Wrong! The correct answer is:", icon="❌")
                    st.info(question["correct_answer"])
                    st.session_state["Wrong"] += 1
            
            answers[f"q{index}"] = r
            st.divider()
    
    st.session_state["Answers"] = answers
    
    if st.session_state["Attempted"] == st.session_state["NumQuestions"]:
        score_percentage = (st.session_state["Correct"] / st.session_state["NumQuestions"]) * 100
        st.balloons()
        st.success(f"Quiz completed! Your score: {score_percentage:.1f}%")

def init_session_variables():
    """Initialize or reset session state variables."""
    if "NumQuestions" not in st.session_state:
        st.session_state["NumQuestions"] = 0
    if "Attempted" not in st.session_state:
        st.session_state["Attempted"] = 0
    if "Correct" not in st.session_state:
        st.session_state["Correct"] = 0
    if "Wrong" not in st.session_state:
        st.session_state["Wrong"] = 0
    if "Submitted" not in st.session_state:
        st.session_state["Submitted"] = False
    if "Answers" not in st.session_state:
        st.session_state["Answers"] = {}
    if "Questions" not in st.session_state:
        st.session_state["Questions"] = None

def reset_quiz():
    """Reset all quiz-related session variables."""
    st.session_state["NumQuestions"] = 0
    st.session_state["Attempted"] = 0
    st.session_state["Correct"] = 0
    st.session_state["Wrong"] = 0
    st.session_state["Submitted"] = False
    st.session_state["Answers"] = {}
    st.session_state["Questions"] = None
    
    # Clear all answered question flags
    keys_to_clear = [key for key in st.session_state.keys() if key.startswith("answered_q")]
    for key in keys_to_clear:
        del st.session_state[key]

def main():
    st.set_page_config(page_title="MCQ Generator", page_icon="📚", layout="wide")
    
    st.title("📚 Smart MCQ Generator")
    st.markdown("""
    This application generates multiple-choice questions based on your input text.
    The questions are generated using natural language processing and vary in complexity.
    """)
    
    init_session_variables()
    
    # Input section
    st.header("Input Text")
    text_input = st.text_area(
        "Enter your text or paste multiple messages (one per line):",
        height=150,
        help="Enter the text from which you want to generate MCQs. Each line will be treated as a separate message."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        num_questions = st.number_input(
            'Number of MCQs to generate',
            min_value=1,
            max_value=10,
            value=5,
            help="Choose how many questions you want to generate"
        )
    
    with col2:
        generate_btn = st.button('Generate MCQs', type="primary")
    
    with col3:
        reset_btn = st.button('Reset Quiz', type="secondary")
    
    if reset_btn:
        reset_quiz()
    
    if generate_btn and text_input:
        messages = [msg.strip() for msg in text_input.split('\n') if msg.strip()]
        
        with st.spinner("Generating questions..."):
            try:
                analyzer = ChatAnalyzer()
                st.session_state["Questions"] = analyzer.generate_mcqs(messages, num_questions)
                st.session_state["NumQuestions"] = num_questions
                st.success("Questions generated successfully!")
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
    
    # Display the quiz if questions are generated
    if st.session_state["Questions"]:
        displayAssignment(st.session_state["Questions"])

if __name__ == "__main__":
    main()