# **OVERVIEW**
I've set up this repository with the intent of opening this project up to the open-source community. The goal of this project is to create an open-source application that will help those affected by Alzheimer's when they're struggling to find words in conversation. 


As many of your are likely aware, a lot of the frustration for those with Alzheimer's early-on is derived from the feelings of annoyance and powerlessness when their train of thought is constantly interrupted by the inability to communicate their thoughts and ideas. Alzheimer's and other forms of dementia are very complex conditions, however--after having vulnerable conversations with multiple individuals struggling with the condition--I have strong reason to believe that the social withdrawal & isolation response is not fully a direct outcome of the condition itself. It seems very likely that this behavioral trend is at least *to some extent* due to the culmination of negative experiences, internal frustration, and feelings of helplessness that are linked to the action of social engagement. To what extent? I can't say, as this is not a scientifically grounded conclusion.


I ran the idea for this sort of program by a dear friend of mine suffering from Alzheimer's, Charles Colman--an incredibly bright man who is making a genuinely heartwarming effort to continue trying to share his time with those around him, regardless of how noticably frustrated it can make him--and his eyes lit up at the thought that there might be something that can help him continue engaging with family and friends, even if it isn't a "cure" by any means. I am not an expert at programming, rather a math nerd, but I know that all of the technology already exists for an application that accomplishes this goal, and was honestly shocked something hasn't already been made with this application in mind. As of early October, 2025, I have created a working prototype, though I relied heavily on help from LLMs to compensate for my lack of expertise & desire to just get a proof of concept working ASAP, and it has many limitations in its ability to reasonably be applied in practical scenarios. I will list current limitations below as a sort of "laundry list" of things that could be massively improved to help get things started. 


I want this project to remain open-source so that it can reasonably help as many people as possible with minimal barriers, but this means I can't afford to do this alone. I recognize that there are many incredible people and communities who contribute to open-source projects, and I hope that this is something you deem worth your time to assist with.

# **Improvements**
Quick list of largest current limitations if you need ideas on where to start:

• The ASR (automatic speech recognition module) takes a while to process speech into text. Something with lower latency at the expense of some level of accuracy would likely be much more practical.

• Currently drawing from a pre-defined set of terms for the response dictionary as a sort of "placeholder" for a more holistic dictionary search. Configuring this is necessary for practical application.

• Current interface is not reasonably functional for and end-user.

• Repository is horrendously organized. If you want to completely restructure this for ease of development, please do ;-;
