import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import StringIO
import math


st.set_page_config(
    page_title="Évaluation de la personnalité DISC :bust_in_silhouette:",
    layout="wide",
    page_icon=":bust_in_silhouette:",
)

# Custom CSS to improve app appearance
st.markdown(
    """
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #184b6a;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #d3d3d3d3;
    }

</style>
""",
    unsafe_allow_html=True,
)

# App Title with custom styling
st.markdown(
    "<h1 style='text-align: center; color: #184b6a;'>Évaluation de la personnalité DISC 👤</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-style: italic;'>Découvrez votre style de personnalité DISC en répondant aux questions ci-dessous.</p>",
    unsafe_allow_html=True,
)

# Ensure session state variables are initialized
if "started" not in st.session_state:
    st.session_state.started = False

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "raw_score" not in st.session_state:
    st.session_state.raw_score = {"D": 0, "I": 0, "S": 0, "C": 0}

if "score" not in st.session_state:
    st.session_state.score = {"D": 0, "I": 0, "S": 0, "C": 0}

# If the user hasn't started the assessment yet
if not st.session_state.started:
    st.markdown(
        """
        ### Bienvenue au test de personnalité DISC

        Le test DISC est un outil qui vous aide à comprendre vos traits de personnalité selon quatre dimensions principales : Dominance (D), Influence (I), Stabilité (S) et Consciencieux (C). En répondant à une série de questions, vous découvrirez votre style de personnalité et comprendrez mieux votre façon d'interagir avec les autres.

        #### Instructions :
        - Vous allez lire une série d'affirmations.
        - Pour chaque affirmation, indiquez votre degré d'accord ou de désaccord à l'aide des options proposées.
        - Les valeurs sont les suivantes : **1** : pas du tout d'accord, **2** : plutôt pas d'accord, **3** : neutre, **4** : plutôt d'accord, **5** : tout à fait d'accord.
        - Une fois le test terminé, vous recevrez votre profil DISC ainsi qu'une analyse détaillée de vos résultats.
        - Lisez attentivement les descriptions, car certaines peuvent sembler similaires mais avoir des significations différentes.

        Cliquez sur "Commencer" pour démarrer le test !
        """
    )

    # Create layout for buttons
    c1, c2, c3 = st.columns([1, 2, 1])

    # Handle the "Let's Begin" button press
    if c1.button("Commencer"):
        st.session_state.started = True
        st.session_state.submitted = False
        st.rerun()  # Rerun the script to move to the next stage

    # File upload option, which is shown after clicking "Upload Previous Results"
    if c3.checkbox("Télécharger les résultats précédents"):
        # Display the file uploader when the button is clicked
        uploaded_file = st.file_uploader(
            "Téléchargez vos résultats JSON précédents", type=["json"]
        )
        if uploaded_file is not None:
            # Read the file as bytes
            bytes_data = uploaded_file.getvalue()

            # Convert bytes to string and then to a StringIO object
            stringio = StringIO(bytes_data.decode("utf-8"))

            # Read the JSON content from the string
            try:
                # Load the JSON content
                uploaded_file_content = json.load(stringio)
                # Assume the uploaded content is the normalized_score
                st.session_state.normalized_score = uploaded_file_content

                # Set session states to move forward
                st.session_state.started = True
                st.session_state.show_results = True
                st.session_state.submitted = True
                st.write("Fichier téléchargé et traité avec succès !")
                st.rerun()  # Rerun to process the results

            except json.JSONDecodeError as e:
                st.error(f"Erreur de décodage JSON: {e}")


# Updated plot function
def create_disc_plot(resultant_angle, resultant_magnitude):
    # Reduce the size of the figure by modifying figsize
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": "polar"}
    )  # Reduce size from 10x10 to 6x6

    # The rest of your plotting code remains unchanged
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.01)

    ax.plot(
        resultant_angle,
        resultant_magnitude,
        "o",
        markersize=24,
        color="#4CAF50",
        label="Votre style DISC",
    )

    categories = ["D", "I", "S", "C"]
    angles = [7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4]
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=14, fontweight="bold")

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=np.pi, color="gray", linestyle="--", alpha=0.7)

    ax.axvline(x=np.pi / 2, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=3 * np.pi / 2, color="gray", linestyle="--", alpha=0.7)

    ax.set_yticklabels([])

    ax.grid(True, alpha=0.3)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#f0f2f6")

    plt.title("Votre profil de style DISC", fontsize=16, fontweight="bold", pad=20)

    return fig


# Function to download JSON data
def get_json_download_link(normalized_score):
    json_str = json.dumps(normalized_score, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="disc_results.json">Télécharger les résultats JSON</a>'
    return href


def get_json_download_button(normalized_score):
    # Convert the normalized score dictionary into a JSON string
    json_str = json.dumps(normalized_score, indent=2)

    # Create a downloadable button using the JSON data
    st.download_button(
        label="Télécharger les résultats JSON",
        data=json_str,
        file_name="disc_results.json",
        mime="application/json",
    )


# Function to download PDF report
def get_pdf_download_link(pdf_buffer):
    # Encode the PDF buffer in base64 for downloading
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="disc_report.pdf">Télécharger le rapport PDF</a>'
    return href


def get_pdf_download_button(pdf_buffer):
    # Encode the PDF buffer in base64
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()

    # Convert base64 PDF into a downloadable button
    st.download_button(
        label="Télécharger le rapport PDF",
        data=pdf_buffer.getvalue(),
        file_name="disc_report.pdf",
        mime="application/pdf",
    )


def describe_style(normalized_score, resultant_angle):
    """
    Détermine le style DISC de l'utilisateur à partir du vecteur résultant (angle et magnitude) sur la roue DISC.

    Arguments:
    normalized_score: Dictionnaire des scores normalisés pour les styles D, I, S et C.
    resultant_angle: Angle résultant en radians.
    resultant_magnitude: Magnitude résultante (intensité de la combinaison des traits).

    Retour:
        Description du style DISC sous forme de chaîne de caractères.        
    """

    # Convert the resultant angle from radians to degrees
    resultant_degrees = math.degrees(resultant_angle)
    if resultant_degrees < 0:
        resultant_degrees += 360  # Convert negative angles to positive

    # Define the angular ranges for main styles and combinations
    style_ranges = {
        # D (Dominance)
        "D": (315, 337.5),
        "DC": (270, 315),
        "DI": (337.5, 360),  # Also covers the 0 degree point
        
        # I (Influence)
        "I": (45, 67.5),
        "ID": (0, 45),
        "IS": (67.5, 90),
        
        # S (Steadiness)
        "S": (135, 157.5),
        "SI": (90, 135),
        "SC": (157.5, 180),
        
        # C (Conscientiousness)
        "C": (225, 247.5),
        "CS": (180, 225),
        "CD": (247.5, 270)
    }

    # Check if all normalized scores are equal (balanced style)
    if all(score == list(normalized_score.values())[0] for score in normalized_score.values()):
        st.markdown("### Style équilibré")
        st.markdown(
            "Vos réponses indiquent une personnalité équilibrée, sans préférence marquée pour un style DISC particulier."
        )
        return "Style équilibré"

    # Determine which range the resultant angle falls into
    for style, (start_angle, end_angle) in style_ranges.items():
        if start_angle <= resultant_degrees < end_angle or (start_angle == 337.5 and resultant_degrees == 0):
            # If it's a combination style, look up the combination description
            description = disc_descriptions["single"][style]
            
            # Display the result
            st.markdown(f"{description['title']}\n\n{description['description']}")
            st.markdown(f"**Points forts:** {description['strengths']}")
            st.markdown(f"**Défis:** {description['challenges']}")
            return f"{description['title']}\n\n{description['description']}\n\nPoints forts: {description['strengths']}\n\nDéfis: {description['challenges']}"
    
    # Default fallback if no match is found
    st.markdown("### Style équilibré")
    st.markdown(
        "Vos réponses indiquent une personnalité équilibrée, sans préférence marquée pour un style DISC particulier."
    )
    return "Style équilibré"



def normalize_scores(scores, questions):
    max_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}
    min_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}

    for q in questions:
        for style in ["D", "I", "S", "C"]:
            mapping = q["mapping"][style]
            if mapping >= 0:
                max_contribution = mapping * 2  # Max when (answer - 3) = +2
                min_contribution = mapping * (-2)  # Min when (answer - 3) = -2
            else:
                max_contribution = mapping * (-2)  # Max when (answer - 3) = -2
                min_contribution = mapping * 2  # Min when (answer - 3) = +2

            max_possible_scores[style] += max_contribution
            min_possible_scores[style] += min_contribution

    print(f"Scores maximum possibles: {max_possible_scores}")
    print(f"Scores minimaux possibles: {min_possible_scores}")

    normalized_scores = {}
    for style in ["D", "I", "S", "C"]:
        # Ensure the raw score is within the possible range
        score = max(min(scores[style], max_possible_scores[style]), min_possible_scores[style])
        score_range = max_possible_scores[style] - min_possible_scores[style]
        if score_range == 0:
            normalized_scores[style] = 50.0  # Neutral score if no variation is possible
        else:
            normalized_scores[style] = ((score - min_possible_scores[style]) / score_range) * 100
            # Ensure the normalized score is within 0 to 100
            normalized_scores[style] = max(0, min(normalized_scores[style], 100))
    return normalized_scores


# Function to create PDF report
def create_pdf_report(normalized_score, relative_percentages, fig, style_description):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Define custom styles for better formatting
    styles.add(ParagraphStyle(name='Justify', alignment=4, leading=12))
    styles.add(ParagraphStyle(name='Heading2Center', parent=styles['Heading2'], alignment=1))
    styles.add(ParagraphStyle(name='BodyTextCenter', parent=styles['BodyText'], alignment=1))
    
    story = []

    # Title
    story.append(Paragraph("Rapport d'évaluation de la personnalité DISC", styles["Title"]))
    story.append(Spacer(1, 20))

    # Add Introduction
    story.append(Paragraph("Merci d'avoir complété le test de personnalité DISC. Ce rapport vous donne un aperçu de votre style de personnalité en fonction de vos réponses.", styles['Justify']))
    story.append(Spacer(1, 20))

    # Add DISC Style Breakdown (Absolute Scores)
    story.append(Paragraph("Votre analyse de style DISC (scores absolus):", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Les scores suivants représentent votre niveau absolu dans chaque style DISC sur une échelle de 0 % à 100 %. Un pourcentage plus élevé indique une tendance plus marquée pour ce style.", styles['Justify']))
    story.append(Spacer(1, 10))

    # Create a table for absolute scores
    data = [['Style', 'Score (0-100%)']]
    for style, score in normalized_score.items():
        data.append([style, f"{score:.2f}%"])

    table = Table(data, hAlign='LEFT', colWidths=[100, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Add DISC Style Breakdown (Relative Percentages)
    story.append(Paragraph("Répartition de votre style DISC (pourcentages relatifs):", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Ces pourcentages représentent la proportion de chaque style DISC par rapport à votre profil de personnalité global. Le total est égal à 100 %.", styles['Justify']))
    story.append(Spacer(1, 10))

    # Create a table for relative percentages
    data = [['Style', 'Relative Percentage']]
    for style, score in relative_percentages.items():
        data.append([style, f"{score:.2f}%"])

    table = Table(data, hAlign='LEFT', colWidths=[100, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Explanation about the difference between Absolute Scores and Relative Percentages
    story.append(Paragraph("Comprendre vos scores:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Les scores absolus indiquent l'intensité avec laquelle vous manifestez chaque style DISC individuellement, sans comparaison avec les autres styles. Un score plus élevé signifie que vous avez tendance à présenter davantage de comportements associés à ce style.\n\n"
        "Les pourcentages relatifs montrent la contribution de chaque style à votre profil de personnalité global par rapport aux autres styles. La somme de ces pourcentages est égale à 100 % et vous aide à comprendre quels styles sont les plus dominants dans votre personnalité.",
        styles['Justify']
    ))
    story.append(Spacer(1, 100))

    # Add the plot as an image
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)
    img = Image(img_buffer, width=400, height=400)
    story.append(Spacer(1, 10))
    story.append(img)
    story.append(Spacer(1, 20))

    # Add personalized style description
    story.append(Paragraph("Description personnalisée du style de disque:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(style_description.replace("###", ""), styles['Justify']))
    story.append(Spacer(1, 50))

    story.append(PageBreak())
    # Add explanation about each DISC style
    story.append(Paragraph("Comprendre tous les styles DISC:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    styles_list = [
        ("Dominance (D):", "Vous êtes généralement direct, axé sur les résultats et affirmé. Vous êtes motivé par les défis et l'obtention de résultats concrets."),
        ("Influence (I):", "Vous êtes généralement extraverti, enthousiaste et optimiste. Vous appréciez les interactions sociales et aimez persuader les autres."),
        ("Stabilité (S):", "Vous êtes souvent patient, solidaire et avez l'esprit d'équipe. Vous accordez une grande importance à la coopération et à l'harmonie dans les relations."),
        ("Consciencieux (C):", "Vous avez tendance à être analytique, précis et attentif aux détails. Vous privilégiez l'exactitude, la qualité et l'expertise.")
    ]
    for title, description in styles_list:
        story.append(Paragraph(f"<b>{title}</b> {description}", styles['Justify']))
        story.append(Spacer(1, 5))
    story.append(Spacer(1, 10))

    # Add final remarks
    story.append(
        Paragraph(
            "N'oubliez pas que chacun possède des aspects des quatre styles, mais la plupart des gens ont tendance à privilégier un ou deux styles principaux. "
            "Votre combinaison unique de styles influence votre façon de communiquer, de prendre des décisions et d'interagir avec les autres.",
            styles['Justify']
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Utilisez cette observation pour améliorer vos relations personnelles et professionnelles en reconnaissant et en appréciant les différents styles chez vous et chez les autres.",
            styles['Justify']
        )
    )

    # Build the PDF
    doc.build(story)

    # Ensure we return the buffer to be used for downloading
    buffer.seek(0)
    return buffer


# If the user has started the test, proceed with the questions
if st.session_state.started:

    # Initialize session state variables for the questions
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

    if "score" not in st.session_state:
        st.session_state.score = {"D": 0, "I": 0, "S": 0, "C": 0}

    if "answers" not in st.session_state:
        st.session_state.answers = {}

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if "questions" not in st.session_state:
        questions = json.load(open("questions.json", "r", encoding="utf-8"))
        random.shuffle(questions)
        st.session_state.questions = questions[:30]  # Use the first 30 questions

    questions_per_page = 1  # Show one question at a time
    total_questions = len(st.session_state.questions)
    total_pages = (
        total_questions + questions_per_page - 1
    ) // questions_per_page  # Ceiling division

    # Load DISC descriptions
    disc_descriptions = json.load(open("disc_descriptions.json", "r", encoding="utf-8"))

    if not st.session_state.show_results:
        start = st.session_state.page_number * questions_per_page
        end = start + questions_per_page

        # Calculate progress
        progress = (st.session_state.page_number) / total_questions

        # Display progress bar outside the form
        st.progress(progress)
        
        with st.form(key=f"form_{st.session_state.page_number}"):
            i = start
            if i < total_questions:
                q = st.session_state.questions[i]
                n = i + 1
                st.markdown(f"#### {n}) {q['question']}")
                options = [
                    "Sélectionnez une option",
                    "1 - Je suis totalement en désaccord",
                    "2 - Je suis plutôt en désaccord",
                    "3 - Je suis neutre",
                    "4 - Je suis plutôt d'accord",
                    "5 - Je suis tout à fait d'accord",
                ]
                selected_option = st.radio(
                    "Choisissez votre réponse",
                    options=options,
                    index=0,
                    key=f"radio_{i}",
                    horizontal=True,
                )
                if st.session_state.page_number < total_pages - 1:
                    submit_button = st.form_submit_button("Suivant")
                else:
                    submit_button = st.form_submit_button("**Afficher mon style DISC**")
            else:
                submit_button = st.form_submit_button("**Afficher mon style DISC**")

        if submit_button:
            if selected_option == "Sélectionnez une option":
                st.warning("Veuillez sélectionner une réponse pour continuer.")
            else:
                # Map the selected option to a score
                score_mapping = {
                    "1 - Je suis totalement en désaccord": 1,
                    "2 - Je suis plutôt en désaccord": 2,
                    "3 - Je suis neutre": 3,
                    "4 - Je suis plutôt d'accord": 4,
                    "5 - Je suis tout à fait d'accord": 5,
                }
                st.session_state.answers[i] = score_mapping[selected_option]
                if st.session_state.page_number < total_pages - 1:
                    st.session_state.page_number += 1
                    st.rerun()
                else:
                    # Set flags to show results and indicate submission
                    st.session_state.show_results = True
                    st.session_state.submitted = False  # Ensure this is reset
                    st.rerun()
    else:
        # After the user has completed the assessment or uploaded results
        if not st.session_state.submitted:
            # Reset the scores before calculating
            st.session_state.score = {"D": 0, "I": 0, "S": 0, "C": 0}
            # Calculate raw scores
            for i in range(total_questions):
                q = st.session_state.questions[i]
                answer = st.session_state.answers[i]
                for style in ["D", "I", "S", "C"]:
                    st.session_state.score[style] += q["mapping"][style] * (answer - 3)
            print(f'Raw score: {st.session_state.score}')
            st.session_state.raw_score = st.session_state.score.copy()

            # Normalize the scores
            normalized_score = normalize_scores(st.session_state.score, st.session_state.questions)
            print(f'Score normalisé: {normalized_score}')
            st.session_state.normalized_score = normalized_score
            st.session_state.submitted = True  # Set to True to avoid recalculation
        
        
        else:
            # Check if normalized_score exists in session_state
            if 'normalized_score' in st.session_state:
                normalized_score = st.session_state.normalized_score
            else:
                # If not, recalculate it from st.session_state.score
                normalized_score = normalize_scores(st.session_state.score, st.session_state.questions)
                st.session_state.normalized_score = normalized_score

        print(f'Score normalisé: {normalized_score}')
        
        # Define the categories and their positions
        categories = ["D", "I", "S", "C"]

        # Angles for the styles
        angles = [7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4]

        # Prepare the values
        values = [normalized_score[cat] for cat in categories]
        
        # Divide Each Normalized Score by 100:
        scaled_scores = {style: score / 100 for style, score in normalized_score.items()}

        # Compute x and y components of the style vectors
        x_components = []
        y_components = []
        for style in categories:
            angle = angles[categories.index(style)]
            magnitude = scaled_scores[style]
            x_components.append(magnitude * np.cos(angle))
            y_components.append(magnitude * np.sin(angle))

        # Sum the components
        total_x = sum(x_components)
        total_y = sum(y_components)

        # Compute the resultant vector
        resultant_magnitude = np.sqrt(total_x**2 + total_y**2)
        resultant_angle = np.arctan2(total_y, total_x)
        
        print(f"Ampleur résultante: {resultant_magnitude}")

        # Ensure the magnitude does not exceed 1
        # resultant_magnitude = min(resultant_magnitude, 1.0)

        # Create the updated plot
        fig = create_disc_plot(resultant_angle, resultant_magnitude)

        # Use Streamlit columns to control the figure width
        col1, col2, col3 = st.columns(
            [1, 2, 1]
        )  # Adjust the middle column width (2/4 of the page width)

        with col2:  # Display the plot in the middle column
            st.pyplot(fig)

        # Personalized Style Descriptions
        st.markdown("## Votre style DISC personnalisé")
        style_description = describe_style(normalized_score, resultant_angle)

        # Display normalized scores with progress bars
        
        ### Here is was using the style usage rather than general style breakdown
        # st.markdown("## Your DISC Style Breakdown")
        # st.markdown("### Style Usage Scores (How much you use each style)")
        # cols = st.columns(4)
        # for idx, (style, score_value) in enumerate(normalized_score.items()):
        #     with cols[idx]:
        #         st.markdown(f"**{style}**")
        #         # Ensure score_value is within 0 to 100
        #         score_value = max(0, min(score_value, 100))
        #         # Adjust the progress bar value to be between 0.0 and 1.0
        #         st.progress(score_value / 100)
        #         # Display the score as a percentage
        #         st.text(f"{score_value:.2f}%")
                
        total_normalized = sum(normalized_score.values())
        relative_percentages = {}
        for style, score in normalized_score.items():
            if total_normalized == 0:
                relative_percentages[style] = 0
            else:
                relative_percentages[style] = (score / total_normalized) * 100
        
        st.markdown("## Analyse de votre style DISC")
        st.write("Pourcentages relatifs")
        cols = st.columns(4)
        for idx, (style, score_value) in enumerate(relative_percentages.items()):
            with cols[idx]:
                # Color by style letter
                if style == "D":
                    color = "#e74c3c"  # red
                elif style == "I":
                    color = "#f39c12"  # orange
                elif style == "S":
                    color = "#2ecc71"  # green
                elif style == "C":
                    color = "#3498db"  # blue
                else:
                    color = "black"

                st.markdown(f"<div style='font-weight:bold; color: {color}; font-size:18px'>{style}</div>", unsafe_allow_html=True)
                # Ensure score_value is within 0 to 100
                score_value = max(0, min(score_value, 100))

                # Render a custom progress bar using HTML so we can control the color per style
                bar_html = f"""
                <div style='background:#e6e6e6; border-radius:8px; padding:3px;'>
                  <div style='width: {score_value}%; background: {color}; height: 14px; border-radius:6px;'></div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)
                # Display the score as a percentage
                st.markdown(f"<div style='margin-top:6px; font-size:14px'>{score_value:.2f}%</div>", unsafe_allow_html=True)
        
        # Download options
        st.markdown("## Téléchargez vos résultats")
        col1, col2 = st.columns(2)
        with col1:
            get_json_download_button(normalized_score)
        with col2:
            pdf_buffer = create_pdf_report(
                            normalized_score=normalized_score,
                            relative_percentages=relative_percentages,
                            fig=fig,
                            style_description=style_description
                        )
            get_pdf_download_button(pdf_buffer)

        # Explanation about DISC styles
        st.markdown("""---""")
        st.markdown(
            """
        ### Comprendre tous les styles DISC

        - **Dominance (D)**: Vous avez tendance à être direct, axé sur les résultats et affirmé.
        - **Influence (I)**: Vous êtes généralement extraverti, enthousiaste et optimiste.
        - **Stabilité (S)**: Vous êtes souvent patient, solidaire et avez l'esprit d'équipe.
        - **Consciencieux (C)**: Vous avez tendance à être analytique, précis et attentif aux détails.

        N'oubliez pas que chacun possède des aspects des quatre styles, mais la plupart des gens ont tendance à privilégier un ou deux styles principaux.
        Votre combinaison unique de styles influence votre façon de communiquer, de prendre des décisions et d'interagir avec les autres.
        """
        )

        if st.button("Restart"):
            st.session_state.pop("page_number")
            st.session_state.pop("score")
            st.session_state.pop("answers")
            st.session_state.pop("show_results")
            st.session_state.pop("questions")
            st.rerun()


st.markdown(
    """
    ---
    <div style="text-align: center;">
        <strong><a href="https://github.com/dzyla/disc-personality-assessment">Source code</a></strong> | Développé par <a href="https://dzyla.com">Dawid Zyla</a>
    </div>
    """,
    unsafe_allow_html=True,
)