from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

class Review(TypedDict):
    key_points: Annotated[str, 'Provide all the key points discussed in this review']
    summary: Annotated[str, 'Generate a brief summary']
    sentiment: Annotated[str, 'Give Positive, Negative, or Neutral']
    rating: Annotated[int, 'Give the rating as an integer out of 5 based on the review']
    pros: Annotated[Optional[list[str]], 'Write down all the pros inside a list']
    cons: Annotated[Optional[list[str]], 'Write down all the cons inside a list']


# Some LLMs provide structured output, some do not
structured_model = llm.with_structured_output(Review)

kannapa = """Kannappa had the potential to be a great folk tale with devotional sentiment, but sadly, it falls flat in many areas.

The first half was honestly a struggle to sit through—poor pacing and a lack of engagement. The final 30 minutes do pick up a bit, especially after Prabhas’s entry, which brings some energy and star power. But even that can only do so much.

Visually, it feels like a throwback to old-school devotional movies, but with modern CGI effects... and not in a good way. The VFX and CGI work are far from believable—at times, it honestly reminded me of something straight out of a Naagin TV series.

They packed the film with big celebrity cameos, but none of them left a lasting impact. Action sequences felt artificial and over-the-top. Songs are forgettable and don’t leave you humming after.

Also, random observation: the main heroine’s running scenes unintentionally turned into comedy for me—couldn’t help but chuckle there.

On top of all that, for a movie that’s supposed to lean devotional, there are a few awkward, steamy scenes between the leads that might make it uncomfortable even if you take your parents along.

My rating?
Honestly, the only reason I’d give it any points is for Prabhas’s cameo. He pulled off his part with charisma and presence."""

review_1 = """The movie promised an exciting story but fell flat from the start.
The pacing was uneven, making it hard to stay engaged.
Special effects looked cheap and unconvincing.
Overall, it was a disappointing watch with little to enjoy."""




result = structured_model.invoke(kannapa)

print(result)
print("Summary:", result['summary'])
