import pandas as pd
from sentence_transformers import SentenceTransformer, util

qa_data = pd.read_csv("medical_qa.csv")

questions = qa_data["question"].tolist()
answers = qa_data["answer"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

question_embeddings = model.encode(questions, convert_to_tensor=True)


# Follow-up questions database
follow_up = {

    "fever": [
        "Do you also have chills?",
        "Are you experiencing body pain?",
        "Do you feel weakness or fatigue?"
    ],

    "headache": [
        "Do you also feel nausea?",
        "Are you sensitive to light?",
        "Do you have dizziness?"
    ],

    "stomach": [
        "Do you have vomiting?",
        "Do you feel nausea?",
        "Did you eat outside food recently?"
    ],

    "cough": [
        "Do you have sore throat?",
        "Do you have breathing difficulty?",
        "Is the cough dry or with mucus?"
    ],

    "chest": [
        "Do you feel pressure in your chest?",
        "Do you feel pain in your left arm?",
        "Are you experiencing shortness of breath?"
    ]
}


def health_chat(user_question):

    q = user_question.lower()

    # Greeting
    if "hi" in q or "hello" in q:
        return "Hello! I am your AI health assistant. Ask me about symptoms, diseases, diet, or prevention."

    # Semantic search answer
    query_embedding = model.encode(user_question, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, question_embeddings)

    best_match = scores.argmax()

    confidence = scores[0][best_match]

    if confidence > 0.45:

        answer = answers[best_match]

        # add follow up questions if symptom detected
        for symptom in follow_up:
            if symptom in q:

                questions_list = follow_up[symptom]

                follow = "\n\nFollow-up questions:\n"

                for f in questions_list:
                    follow += "• " + f + "\n"

                return answer + follow

        return answer

    return "I'm not sure about that. Please ask a health-related question."