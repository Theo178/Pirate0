import requests
import json
import matplotlib.pyplot as plt
import numpy as np

# API Endpoints
SCENE_URL = "http://localhost:8000/api/analyze-scene"
ARC_URL = "http://localhost:8000/api/analyze-character-arc"

scene_text = """
INT. CAMPUS BAR - NIGHT

MARK ZUCKERBERG is a sweet looking 19 year old whose lack of
any physically intimidating attributes masks a very
complicated and dangerous anger. He has trouble making eye
contact and sometimes it's hard to tell if he's talking to
you or to himself.

ERICA ALBRIGHT is 19 as well and she has a girl-next-door
face that makes her easier to fall for than the girls in
Victoria's Secret catalogs.

The scene is stark and simple.

MARK
Did you know there are more people
with genius IQs living in China than
there are people of any kind living
in the United States?

ERICA
That can't be true.

MARK
It is true.

ERICA
What would the difference be?

MARK
First of all, there are a lot of
people who live in China. But here's
my question. How do you distinguish
yourself in a population of people
who all got 1600 on their SATs?

ERICA
I didn't know they take SATs in China.

MARK
I wasn't talking about China anymore,
I was talking about me.

ERICA
You got 1600?

MARK
Yes. I could sing in an a Capella
group, but I can't sing.

ERICA
Does that mean you actually got 1600?

MARK
350,000 people.

ERICA
Wait--

MARK
--I'm talking about China. 350,000
people with genius IQs...

ERICA
(pause)
You could row crew.

MARK
I can't row crew. I'm 5'8", 150 pounds,
and I just got into final clubs.
"""

payload = {
    "scene_text": scene_text
}

print(f"Sending Scene Analysis request to {SCENE_URL}...")
try:
    # 1. Standard Analysis
    response = requests.post(SCENE_URL, json=payload)
    if response.status_code == 200:
        print("\n--- Visual Scene Analysis ---")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nScene Analysis Error: {response.status_code}")
        print(response.text)

    # 2. Character Arc Analysis
    print(f"\nSending Character Arc request to {ARC_URL}...")
    arc_response = requests.post(ARC_URL, json=payload)
    
    if arc_response.status_code == 200:
        arc_data = arc_response.json()
        print("\n--- Character Arc Data ---")
        print(json.dumps(arc_data, indent=2))
        
        # 3. Generate Graph
        if isinstance(arc_data, list) and len(arc_data) > 0:
            print("\nGenerating graph...")
            characters = sorted(list(set(d['character'] for d in arc_data)))
            
            plt.figure(figsize=(12, 6))
            
            for char in characters:
                # Filter out records missing required keys (e.g. if LLM output was truncated)
                char_data = [d for d in arc_data if d['character'] == char and 'intensity' in d and 'line' in d]
                
                if not char_data:
                    continue

                x_vals = range(len(char_data))
                intensity_vals = [d['intensity'] for d in char_data]
                lines = [d['line'][:20] + "..." for d in char_data] # Snippets
                
                plt.plot(x_vals, intensity_vals, marker='o', linestyle='-', linewidth=2, label=char)
                
                # Annotate points
                # for i, txt in enumerate(lines):
                #     plt.annotate(txt, (x_vals[i], intensity_vals[i]), fontsize=8, alpha=0.7)

            plt.title('Character Emotional Intensity Arc')
            plt.xlabel('Progression (Line by Line)')
            plt.ylabel('Emotional Intensity (1-10)')
            plt.ylim(0, 11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # Save graph
            output_file = 'character_arc_graph.png'
            plt.savefig(output_file)
            print(f"\nGraph saved to: {output_file}")
            
    else:
        print(f"\nCharacter Arc Error: {arc_response.status_code}")
        print(arc_response.text)

    # 4. Generate Storyboard
    STORYBOARD_URL = "http://localhost:8000/api/generate-storyboard"
    print(f"\nSending Storyboard Generation request to {STORYBOARD_URL}...")
    sb_response = requests.post(STORYBOARD_URL, json=payload)
    
    if sb_response.status_code == 200:
        sb_data = sb_response.json()
        print("\n--- Storyboard Data ---")
        print(f"Prompt: {sb_data['image_prompt']}")
        print(f"URL: {sb_data['image_url']}")
        
        # Download Image
        print("\nDownloading storyboard image...")
        img_response = requests.get(sb_data['image_url'])
        if img_response.status_code == 200:
             with open("storyboard.jpg", "wb") as f:
                 f.write(img_response.content)
             print("Saved image to 'storyboard.jpg'")
        else:
             print("Failed to download image")
    else:
        print(f"\nStoryboard Error: {sb_response.status_code}")
        print(sb_response.text)

    # 5. Generate Sequence Sketches
    SEQUENCE_URL = "http://localhost:8000/api/generate-sequence"
    print(f"\nSending Sequence Generation request to {SEQUENCE_URL}...")
    seq_response = requests.post(SEQUENCE_URL, json=payload)
    
    if seq_response.status_code == 200:
        seq_data = seq_response.json()
        print(f"\n--- Sequence Data ({len(seq_data)} shots) ---")
        
        for i, shot in enumerate(seq_data):
            print(f"\nShot {shot.get('shot_id')} ({shot.get('type')}):")
            print(f"Prompt: {shot.get('prompt')[:100]}...")
            print(f"URL: {shot.get('url')}")
            
            # Download Image
            img_res = requests.get(shot.get('url'))
            if img_res.status_code == 200:
                 fname = f"sketch_shot_{i+1}.jpg"
                 with open(fname, "wb") as f:
                     f.write(img_res.content)
                 print(f"Saved sketch to '{fname}'")

    else:
        print(f"\nSequence Error: {seq_response.status_code}")
        print(seq_response.text)

except Exception as e:
    print(f"\nFailed to connect: {e}")
    print("Make sure the backend server is running on localhost:8000")
