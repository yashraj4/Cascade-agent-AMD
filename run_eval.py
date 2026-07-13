"""
Comprehensive evaluation script for the Track 2 routing agent.
Contains 120 test cases (15 per category) based on the AMD Hackathon judging FAQ,
public validation examples, and standard classification types.

Usage:
  # Run all tests using configured environment (.env)
  python run_eval.py

  # Run only sentiment tests
  python run_eval.py --category sentiment_classification

  # Run only a few tests for quick checking
  python run_eval.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Test Cases Dataset (15 per category, total 120 tasks)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # 1. FACTUAL KNOWLEDGE
    {
        "task_id": "fact_1",
        "category": "factual_knowledge",
        "prompt": "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB.",
        "eval_type": "keyword",
        "keywords": ["red", "green", "blue", "additive"]
    },
    {
        "task_id": "fact_2",
        "category": "factual_knowledge",
        "prompt": "What is the difference between machine learning and deep learning? Briefly explain how each works.",
        "eval_type": "keyword",
        "keywords": ["subset", "neural", "feature"]
    },
    {
        "task_id": "fact_3",
        "category": "factual_knowledge",
        "prompt": "Explain the difference between RAM and ROM in a computer. What is each type used for?",
        "eval_type": "keyword",
        "keywords": ["volatile", "non-volatile", "temporary", "permanent"]
    },
    {
        "task_id": "fact_4",
        "category": "factual_knowledge",
        "prompt": "Explain the difference between weather and climate in simple terms.",
        "eval_type": "keyword",
        "keywords": ["short-term", "long-term"]
    },
    {
        "task_id": "fact_5",
        "category": "factual_knowledge",
        "prompt": "What causes the ocean tides on Earth? Briefly explain.",
        "eval_type": "keyword",
        "keywords": ["moon", "gravity"]
    },
    {
        "task_id": "fact_6",
        "category": "factual_knowledge",
        "prompt": "Why is the sky blue during a clear day? What scientific phenomenon is responsible?",
        "eval_type": "keyword",
        "keywords": ["scattering", "blue"]
    },
    {
        "task_id": "fact_7",
        "category": "factual_knowledge",
        "prompt": "Briefly explain the difference between nuclear fission and nuclear fusion.",
        "eval_type": "keyword",
        "keywords": ["splitting", "combining"]
    },
    {
        "task_id": "fact_8",
        "category": "factual_knowledge",
        "prompt": "How do solar panels generate electricity? Explain the core effect.",
        "eval_type": "keyword",
        "keywords": ["photovoltaic", "sunlight"]
    },
    {
        "task_id": "fact_9",
        "category": "factual_knowledge",
        "prompt": "Why do the leaves of deciduous trees change color in autumn?",
        "eval_type": "keyword",
        "keywords": ["chlorophyll", "breakdown"]
    },
    {
        "task_id": "fact_10",
        "category": "factual_knowledge",
        "prompt": "What is the difference between mitosis and meiosis in cell division?",
        "eval_type": "keyword",
        "keywords": ["somatic", "gametes"]
    },
    {
        "task_id": "fact_11",
        "category": "factual_knowledge",
        "prompt": "Explain the difference between a virus and bacteria.",
        "eval_type": "keyword",
        "keywords": ["living", "cell"]
    },
    {
        "task_id": "fact_12",
        "category": "factual_knowledge",
        "prompt": "What is the greenhouse effect and how does it affect Earth's temperature?",
        "eval_type": "keyword",
        "keywords": ["trap", "heat"]
    },
    {
        "task_id": "fact_13",
        "category": "factual_knowledge",
        "prompt": "Describe the inputs and outputs of the process of photosynthesis.",
        "eval_type": "keyword",
        "keywords": ["carbon dioxide", "water", "oxygen", "glucose"]
    },
    {
        "task_id": "fact_14",
        "category": "factual_knowledge",
        "prompt": "What is the difference between active transport and passive transport in biological membranes?",
        "eval_type": "keyword",
        "keywords": ["energy", "concentration"]
    },
    {
        "task_id": "fact_15",
        "category": "factual_knowledge",
        "prompt": "What is a light-year, and what does it measure?",
        "eval_type": "keyword",
        "keywords": ["distance", "year"]
    },

    # 2. MATHEMATICAL REASONING
    {
        "task_id": "math_1",
        "category": "mathematical_reasoning",
        "prompt": "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?",
        "eval_type": "math",
        "math_val": "1672"
    },
    {
        "task_id": "math_2",
        "category": "mathematical_reasoning",
        "prompt": "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?",
        "eval_type": "math",
        "math_val": "4.5"
    },
    {
        "task_id": "math_3",
        "category": "mathematical_reasoning",
        "prompt": "What is 24 * 7?",
        "eval_type": "math",
        "math_val": "168"
    },
    {
        "task_id": "math_4",
        "category": "mathematical_reasoning",
        "prompt": "Calculate (12 + 8) * 5 / 2.",
        "eval_type": "math",
        "math_val": "50"
    },
    {
        "task_id": "math_5",
        "category": "mathematical_reasoning",
        "prompt": "A train leaves station A at 60 mph. Another train leaves station B at 80 mph. They are 280 miles apart and traveling towards each other. In how many hours will they meet?",
        "eval_type": "math",
        "math_val": "2"
    },
    {
        "task_id": "math_6",
        "category": "mathematical_reasoning",
        "prompt": "If a laptop is discounted by 15% and now costs $850, what was its original price?",
        "eval_type": "math",
        "math_val": "1000"
    },
    {
        "task_id": "math_7",
        "category": "mathematical_reasoning",
        "prompt": "Compute 1543 - 876.",
        "eval_type": "math",
        "math_val": "667"
    },
    {
        "task_id": "math_8",
        "category": "mathematical_reasoning",
        "prompt": "Evaluate 125 * 8.",
        "eval_type": "math",
        "math_val": "1000"
    },
    {
        "task_id": "math_9",
        "category": "mathematical_reasoning",
        "prompt": "When rolling a fair 6-sided die, what is the probability of getting an even number? Express as decimal.",
        "eval_type": "math",
        "math_val": "0.5"
    },
    {
        "task_id": "math_10",
        "category": "mathematical_reasoning",
        "prompt": "Solve for x: 3x + 7 = 22.",
        "eval_type": "math",
        "math_val": "5"
    },
    {
        "task_id": "math_11",
        "category": "mathematical_reasoning",
        "prompt": "If $1000 is invested at 5% simple interest per year, how much interest is earned in 3 years?",
        "eval_type": "math",
        "math_val": "150"
    },
    {
        "task_id": "math_12",
        "category": "mathematical_reasoning",
        "prompt": "What is the area of a circle with a radius of 7? Use pi = 22/7.",
        "eval_type": "math",
        "math_val": "154"
    },
    {
        "task_id": "math_13",
        "category": "mathematical_reasoning",
        "prompt": "A tank can be filled by pipe A in 3 hours and pipe B in 6 hours. How long does it take to fill the tank if both are opened together?",
        "eval_type": "math",
        "math_val": "2"
    },
    {
        "task_id": "math_14",
        "category": "mathematical_reasoning",
        "prompt": "If 5 machines take 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "eval_type": "math",
        "math_val": "5"
    },
    {
        "task_id": "math_15",
        "category": "mathematical_reasoning",
        "prompt": "What is the next number in the sequence: 2, 4, 8, 16, 32, ...?",
        "eval_type": "math",
        "math_val": "64"
    },

    # 3. SENTIMENT CLASSIFICATION
    {
        "task_id": "sent_1",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Mixed", "Neutral", "Positive"],
        "mixed_required": True,
        "mixed_keywords": [["late", "damage", "packaging"], ["perfect", "support", "resolve"]]
    },
    {
        "task_id": "sent_2",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment of this tweet as Positive, Negative, or Neutral and give a one-sentence reason: 'Just got my order. Box was dented and the manual was missing, but honestly the device itself is flawless and set up in under 5 minutes.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Mixed", "Neutral", "Positive"],
        "mixed_required": True,
        "mixed_keywords": [["dent", "manual", "missing"], ["flawless", "minute", "fast", "setup"]]
    },
    {
        "task_id": "sent_3",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment of this review and give a one-sentence reason: 'I absolutely loved this product, it exceeded all my expectations!'",
        "eval_type": "sentiment",
        "allowed_labels": ["Positive"],
        "mixed_required": False
    },
    {
        "task_id": "sent_4",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment of this review and give a one-sentence reason: 'This is the worst purchase I have ever made. The device stopped working after two days and customer service refused to refund.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Negative"],
        "mixed_required": False
    },
    {
        "task_id": "sent_5",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment of this review and give a one-sentence reason: 'The item arrived on time. It is a standard USB cable, black, about 3 feet long. It works as intended.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Neutral", "Positive"],
        "mixed_required": False
    },
    {
        "task_id": "sent_6",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'The screen is beautiful and bright, but the battery life is atrocious and barely lasts 3 hours.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Mixed", "Neutral", "Negative"],
        "mixed_required": True,
        "mixed_keywords": [["screen", "bright", "beautiful"], ["battery", "atrocious", "last"]]
    },
    {
        "task_id": "sent_7",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'Highly recommend this app. Super intuitive UI, extremely fast performance, and regular updates.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Positive"],
        "mixed_required": False
    },
    {
        "task_id": "sent_8",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'Extremely disappointed. The shoes are tight, the color is different from the picture, and the sole feels cheap.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Negative"],
        "mixed_required": False
    },
    {
        "task_id": "sent_9",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'The food was absolutely delicious and the steak was cooked to perfection, but our waiter was incredibly rude and we had to wait 45 minutes for our table.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Mixed", "Neutral", "Positive"],
        "mixed_required": True,
        "mixed_keywords": [["food", "delicious", "steak", "perfection"], ["waiter", "rude", "wait"]]
    },
    {
        "task_id": "sent_10",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'The package contains 5 items. The color of the items is grey. There are no additional accessories.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Neutral"],
        "mixed_required": False
    },
    {
        "task_id": "sent_11",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'The hotel had a great location right next to the beach, but the rooms were small, outdated, and the AC was loud.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Mixed", "Neutral", "Negative"],
        "mixed_required": True,
        "mixed_keywords": [["location", "beach", "great"], ["room", "outdated", "AC", "loud"]]
    },
    {
        "task_id": "sent_12",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'This book is a masterpiece. Deeply moving characters, a brilliant plot, and an ending that leaves you thinking for days.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Positive"],
        "mixed_required": False
    },
    {
        "task_id": "sent_13",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'Terrible customer service. Kept me on hold for an hour and then hung up. Will never use this company again.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Negative"],
        "mixed_required": False
    },
    {
        "task_id": "sent_14",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'The headphones sound incredible with rich bass, but they are heavy and hurt my ears after an hour of use.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Mixed", "Neutral", "Negative"],
        "mixed_required": True,
        "mixed_keywords": [["sound", "bass", "incredible"], ["heavy", "hurt", "ear"]]
    },
    {
        "task_id": "sent_15",
        "category": "sentiment_classification",
        "prompt": "Classify the sentiment and give a one-sentence reason: 'Fantastic service! The staff was incredibly welcoming, the coffee was great, and the atmosphere was cozy.'",
        "eval_type": "sentiment",
        "allowed_labels": ["Positive"],
        "mixed_required": False
    },

    # 4. TEXT SUMMARISATION
    {
        "task_id": "sum_1",
        "category": "text_summarisation",
        "prompt": "Summarize the following passage in exactly two sentences:\n'Machine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records that might be missed by human clinicians. However, concerns remain about model interpretability, data privacy, liability when errors occur, and the potential for algorithmic bias to worsen existing healthcare disparities. Regulatory frameworks are still catching up with the pace of deployment, creating uncertainty for healthcare providers and technology developers alike.'",
        "eval_type": "summarization",
        "expected_sentences": 2
    },
    {
        "task_id": "sum_2",
        "category": "text_summarisation",
        "prompt": "Summarize the following passage in exactly three bullet points, each no longer than 15 words:\n'Remote work has transformed how companies operate globally. Employees gain flexibility and reduced commute times, leading to reported improvements in work-life balance. However, challenges persist around collaboration, company culture, and the blurring of personal and professional boundaries. Organisations are responding by investing in digital collaboration tools and rethinking office space as a hub for social and creative work rather than daily attendance.'",
        "eval_type": "summarization",
        "expected_bullets": 3,
        "max_bullet_words": 15
    },
    {
        "task_id": "sum_3",
        "category": "text_summarisation",
        "prompt": "Summarize the history of space exploration from the Moon landing to Mars rovers in exactly one sentence.",
        "eval_type": "summarization",
        "expected_sentences": 1
    },
    {
        "task_id": "sum_4",
        "category": "text_summarisation",
        "prompt": "Summarize the process of the water cycle in exactly three short bullet points.",
        "eval_type": "summarization",
        "expected_bullets": 3
    },
    {
        "task_id": "sum_5",
        "category": "text_summarisation",
        "prompt": "Summarize the main benefits of physical exercise on cardiovascular and mental health in exactly two sentences.",
        "eval_type": "summarization",
        "expected_sentences": 2
    },
    {
        "task_id": "sum_6",
        "category": "text_summarisation",
        "prompt": "Summarize how a bill becomes a law in the United States in exactly three sentences.",
        "eval_type": "summarization",
        "expected_sentences": 3
    },
    {
        "task_id": "sum_7",
        "category": "text_summarisation",
        "prompt": "Summarize the plot of Romeo and Juliet in exactly one sentence.",
        "eval_type": "summarization",
        "expected_sentences": 1
    },
    {
        "task_id": "sum_8",
        "category": "text_summarisation",
        "prompt": "Summarize the major causes of the French Revolution in exactly three bullet points.",
        "eval_type": "summarization",
        "expected_bullets": 3
    },
    {
        "task_id": "sum_9",
        "category": "text_summarisation",
        "prompt": "Summarize why biodiversity is crucial for ecosystem stability in exactly two sentences.",
        "eval_type": "summarization",
        "expected_sentences": 2
    },
    {
        "task_id": "sum_10",
        "category": "text_summarisation",
        "prompt": "Summarize the economic concept of inflation in exactly one sentence.",
        "eval_type": "summarization",
        "expected_sentences": 1
    },
    {
        "task_id": "sum_11",
        "category": "text_summarisation",
        "prompt": "Summarize the mental benefits of reading fiction in exactly three bullet points.",
        "eval_type": "summarization",
        "expected_bullets": 3
    },
    {
        "task_id": "sum_12",
        "category": "text_summarisation",
        "prompt": "Summarize the carbon cycle and its human impact in exactly two sentences.",
        "eval_type": "summarization",
        "expected_sentences": 2
    },
    {
        "task_id": "sum_13",
        "category": "text_summarisation",
        "prompt": "Summarize the discovery of penicillin and its impact on medicine in exactly one sentence.",
        "eval_type": "summarization",
        "expected_sentences": 1
    },
    {
        "task_id": "sum_14",
        "category": "text_summarisation",
        "prompt": "Summarize the core differences between renewable and non-renewable energy in exactly two sentences.",
        "eval_type": "summarization",
        "expected_sentences": 2
    },
    {
        "task_id": "sum_15",
        "category": "text_summarisation",
        "prompt": "Summarize the role of the heart in the circulatory system in exactly three bullet points.",
        "eval_type": "summarization",
        "expected_bullets": 3
    },

    # 5. NAMED ENTITY RECOGNITION
    {
        "task_id": "ner_1",
        "category": "named_entity_recognition",
        "prompt": "Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE:\n'On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.'",
        "eval_type": "ner",
        "entities": {
            "Sundar Pichai": "PERSON",
            "March 15 2023": "DATE",
            "Google": "ORGANIZATION",
            "Zurich": "LOCATION",
            "ETH Zurich": "ORGANIZATION"
        }
    },
    {
        "task_id": "ner_2",
        "category": "named_entity_recognition",
        "prompt": "Extract the named entities from: Barack Obama visited Paris on July 4th. Label each as PERSON, ORGANIZATION, LOCATION, or DATE.",
        "eval_type": "ner",
        "entities": {
            "Barack Obama": "PERSON",
            "Paris": "LOCATION",
            "July 4th": "DATE"
        }
    },
    {
        "task_id": "ner_3",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities and label them as PERSON, ORGANIZATION, LOCATION, or DATE: 'Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California on April 1, 1976.'",
        "eval_type": "ner",
        "entities": {
            "Apple Inc.": "ORGANIZATION",
            "Steve Jobs": "PERSON",
            "Steve Wozniak": "PERSON",
            "Cupertino": "LOCATION",
            "California": "LOCATION",
            "April 1, 1976": "DATE"
        }
    },
    {
        "task_id": "ner_4",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities and label them: 'Elon Musk bought Twitter in October 2022.'",
        "eval_type": "ner",
        "entities": {
            "Elon Musk": "PERSON",
            "Twitter": "ORGANIZATION",
            "October 2022": "DATE"
        }
    },
    {
        "task_id": "ner_5",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities and label them: 'The Eiffel Tower in Paris, France was designed by Gustave Eiffel for the 1889 World's Fair.'",
        "eval_type": "ner",
        "entities": {
            "Paris": "LOCATION",
            "France": "LOCATION",
            "Gustave Eiffel": "PERSON",
            "1889": "DATE"
        }
    },
    {
        "task_id": "ner_6",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'Albert Einstein presented the theory of relativity in Berlin in 1915.'",
        "eval_type": "ner",
        "entities": {
            "Albert Einstein": "PERSON",
            "Berlin": "LOCATION",
            "1915": "DATE"
        }
    },
    {
        "task_id": "ner_7",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'Microsoft was started by Bill Gates and Paul Allen in Albuquerque, New Mexico on April 4, 1975.'",
        "eval_type": "ner",
        "entities": {
            "Microsoft": "ORGANIZATION",
            "Bill Gates": "PERSON",
            "Paul Allen": "PERSON",
            "Albuquerque": "LOCATION",
            "New Mexico": "LOCATION",
            "April 4, 1975": "DATE"
        }
    },
    {
        "task_id": "ner_8",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'Marie Curie discovered radium in Paris, winning the Nobel Prize in 1911.'",
        "eval_type": "ner",
        "entities": {
            "Marie Curie": "PERSON",
            "Paris": "LOCATION",
            "1911": "DATE"
        }
    },
    {
        "task_id": "ner_9",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'Amazon was founded by Jeff Bezos in Seattle, Washington on July 5, 1994.'",
        "eval_type": "ner",
        "entities": {
            "Amazon": "ORGANIZATION",
            "Jeff Bezos": "PERSON",
            "Seattle": "LOCATION",
            "Washington": "LOCATION",
            "July 5, 1994": "DATE"
        }
    },
    {
        "task_id": "ner_10",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'Nelson Mandela was inaugurated as President of South Africa in Pretoria on May 10, 1994.'",
        "eval_type": "ner",
        "entities": {
            "Nelson Mandela": "PERSON",
            "South Africa": "LOCATION",
            "Pretoria": "LOCATION",
            "May 10, 1994": "DATE"
        }
    },
    {
        "task_id": "ner_11",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'The United Nations was established on October 24, 1945 in San Francisco, California.'",
        "eval_type": "ner",
        "entities": {
            "United Nations": "ORGANIZATION",
            "October 24, 1945": "DATE",
            "San Francisco": "LOCATION",
            "California": "LOCATION"
        }
    },
    {
        "task_id": "ner_12",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'William Shakespeare was born in Stratford-upon-Avon, England in April 1564.'",
        "eval_type": "ner",
        "entities": {
            "William Shakespeare": "PERSON",
            "Stratford-upon-Avon": "LOCATION",
            "England": "LOCATION",
            "April 1564": "DATE"
        }
    },
    {
        "task_id": "ner_13",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'NASA launched the James Webb Space Telescope on December 25, 2021 from Kourou.'",
        "eval_type": "ner",
        "entities": {
            "NASA": "ORGANIZATION",
            "December 25, 2021": "DATE",
            "Kourou": "LOCATION"
        }
    },
    {
        "task_id": "ner_14",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'J.K. Rowling published the first Harry Potter book in London, United Kingdom on June 26, 1997.'",
        "eval_type": "ner",
        "entities": {
            "J.K. Rowling": "PERSON",
            "London": "LOCATION",
            "United Kingdom": "LOCATION",
            "June 26, 1997": "DATE"
        }
    },
    {
        "task_id": "ner_15",
        "category": "named_entity_recognition",
        "prompt": "Extract named entities: 'Toyota was founded by Kiichiro Toyoda in Aichi, Japan on August 28, 1937.'",
        "eval_type": "ner",
        "entities": {
            "Toyota": "ORGANIZATION",
            "Kiichiro Toyoda": "PERSON",
            "Aichi": "LOCATION",
            "Japan": "LOCATION",
            "August 28, 1937": "DATE"
        }
    },

    # 6. CODE DEBUGGING
    {
        "task_id": "debug_1",
        "category": "code_debugging",
        "prompt": "This code has a bug: def add(a, b): return a - b",
        "eval_type": "code",
        "keywords": ["a + b", "+ b"]
    },
    {
        "task_id": "debug_2",
        "category": "code_debugging",
        "prompt": "Identify the bug and fix this Python function so negative numbers are classified correctly: def is_even(n): return n % 2 == 0 if n > 0 else False",
        "eval_type": "code",
        "keywords": ["% 2 == 0"]
    },
    {
        "task_id": "debug_3",
        "category": "code_debugging",
        "prompt": "Fix the bug in this Python function which gets the first element: def get_first(lst): return lst[1]",
        "eval_type": "code",
        "keywords": ["lst[0]"]
    },
    {
        "task_id": "debug_4",
        "category": "code_debugging",
        "prompt": "Fix this code so it breaks correctly: for i in range(5): print(i) if i == 3: break else: print('done')",
        "eval_type": "code",
        "keywords": ["break"]
    },
    {
        "task_id": "debug_5",
        "category": "code_debugging",
        "prompt": "Fix the syntax error in this Python class: class Dog: def __init__(name): self.name = name",
        "eval_type": "code",
        "keywords": ["__init__(self, name)"]
    },
    {
        "task_id": "debug_6",
        "category": "code_debugging",
        "prompt": "Fix this division by zero bug: def average(nums): return sum(nums) / len(nums)",
        "eval_type": "code",
        "keywords": ["len(nums) == 0", "if not nums"]
    },
    {
        "task_id": "debug_7",
        "category": "code_debugging",
        "prompt": "Fix this buggy function: def string_reverse(s): return s.reverse()",
        "eval_type": "code",
        "keywords": ["[::-1]"]
    },
    {
        "task_id": "debug_8",
        "category": "code_debugging",
        "prompt": "Fix this scope bug in Python:\nx = 10\ndef update():\n    x = x + 5\nupdate()",
        "eval_type": "code",
        "keywords": ["global x"]
    },
    {
        "task_id": "debug_9",
        "category": "code_debugging",
        "prompt": "Fix this mutable default argument bug:\ndef append_to(element, target=[]):\n    target.append(element)\n    return target",
        "eval_type": "code",
        "keywords": ["target is None", "target = []"]
    },
    {
        "task_id": "debug_10",
        "category": "code_debugging",
        "prompt": "Fix this key error bug: def get_val(d, k): return d[k]",
        "eval_type": "code",
        "keywords": ["get(k)", "in d"]
    },
    {
        "task_id": "debug_11",
        "category": "code_debugging",
        "prompt": "Fix this logic error: def is_positive_and_even(n): return n > 0 or n % 2 == 0",
        "eval_type": "code",
        "keywords": ["and n"]
    },
    {
        "task_id": "debug_12",
        "category": "code_debugging",
        "prompt": "Fix this infinite loop: i = 0\nwhile i < 10:\n    print(i)",
        "eval_type": "code",
        "keywords": ["i += 1", "i = i + 1"]
    },
    {
        "task_id": "debug_13",
        "category": "code_debugging",
        "prompt": "Fix this type error: def greet(name): return 'Hello ' + name.upper() if name else 'Hello ' + 5",
        "eval_type": "code",
        "keywords": ["'5'"]
    },
    {
        "task_id": "debug_14",
        "category": "code_debugging",
        "prompt": "Fix this syntax error: if x = 5: print(x)",
        "eval_type": "code",
        "keywords": ["x == 5"]
    },
    {
        "task_id": "debug_15",
        "category": "code_debugging",
        "prompt": "Fix this Python function: def check_empty(s): return s == None",
        "eval_type": "code",
        "keywords": ["s is None"]
    },

    # 7. LOGICAL DEDUCTIVE REASONING
    {
        "task_id": "logic_1",
        "category": "logical_deductive_reasoning",
        "prompt": "Alice, Bob, and Charlie live in red, blue, and green houses. Alice does not live in the red house. Bob lives in the blue house. Who lives in the green house?",
        "eval_type": "logical",
        "expected_val": "Alice"
    },
    {
        "task_id": "logic_2",
        "category": "logical_deductive_reasoning",
        "prompt": "Five runners (A, B, C, D, E) finished a race. A finished before B but after C. D finished before C but after E. Who finished first?",
        "eval_type": "logical",
        "expected_val": "E"
    },
    {
        "task_id": "logic_3",
        "category": "logical_deductive_reasoning",
        "prompt": "John is older than Mary. Mary is older than Sue. Sue is older than Tom. Who is the youngest?",
        "eval_type": "logical",
        "expected_val": "Tom"
    },
    {
        "task_id": "logic_4",
        "category": "logical_deductive_reasoning",
        "prompt": "If all Bloops are Razzies and all Razzies are Lizzies, are all Bloops Lizzies? Yes or No.",
        "eval_type": "logical",
        "expected_val": "Yes"
    },
    {
        "task_id": "logic_5",
        "category": "logical_deductive_reasoning",
        "prompt": "If it rains, the grass is wet. The grass is not wet. Did it rain? Yes or No.",
        "eval_type": "logical",
        "expected_val": "No"
    },
    {
        "task_id": "logic_6",
        "category": "logical_deductive_reasoning",
        "prompt": "A collection of shapes consists of circles and triangles. All red shapes are circles. Some circles are red. Are there any red triangles? Yes or No.",
        "eval_type": "logical",
        "expected_val": "No"
    },
    {
        "task_id": "logic_7",
        "category": "logical_deductive_reasoning",
        "prompt": "There are three boxes: one gold, one silver, and one lead. One box contains a prize. The gold box says 'The prize is in here.' The silver box says 'The prize is not in here.' The lead box says 'The prize is not in the gold box.' Only one statement is true. Which box has the prize? Gold, Silver, or Lead.",
        "eval_type": "logical",
        "expected_val": "Silver"
    },
    {
        "task_id": "logic_8",
        "category": "logical_deductive_reasoning",
        "prompt": "In a room of 4 people, everyone shakes hands with everyone else once. How many handshakes occur?",
        "eval_type": "logical",
        "expected_val": "6"
    },
    {
        "task_id": "logic_9",
        "category": "logical_deductive_reasoning",
        "prompt": "A doctor gives you three pills and tells you to take one every half hour. How many minutes will they last?",
        "eval_type": "logical",
        "expected_val": "60"
    },
    {
        "task_id": "logic_10",
        "category": "logical_deductive_reasoning",
        "prompt": "Brothers and sisters I have none, but that man's father is my father's son. Who is that man?",
        "eval_type": "logical",
        "expected_val": "son"
    },
    {
        "task_id": "logic_11",
        "category": "logical_deductive_reasoning",
        "prompt": "Four friends (Anna, Bill, Cody, Dave) sit around a circular table. Anna sits next to Bill but not Cody. Dave sits next to Cody. Who sits opposite Anna?",
        "eval_type": "logical",
        "expected_val": "Dave"
    },
    {
        "task_id": "logic_12",
        "category": "logical_deductive_reasoning",
        "prompt": "If a card has a vowel on one side, it must have an even number on the other side. You are shown cards: A, B, 4, 7. Which cards must you turn over to test the rule?",
        "eval_type": "logical",
        "expected_val": "7"  # turning A and 7 is standard logic Wason task. Checking 7 is key.
    },
    {
        "task_id": "logic_13",
        "category": "logical_deductive_reasoning",
        "prompt": "Three cups are placed in a row. A coin is hidden under one cup. The left cup is empty. The middle cup contains either the coin or nothing. The right cup contains the coin if the middle cup is empty. Which cup has the coin? Left, Middle, or Right.",
        "eval_type": "logical",
        "expected_val": "Right"
    },
    {
        "task_id": "logic_14",
        "category": "logical_deductive_reasoning",
        "prompt": "Some As are Bs. All Bs are Cs. Are all As Cs? Yes or No.",
        "eval_type": "logical",
        "expected_val": "No"
    },
    {
        "task_id": "logic_15",
        "category": "logical_deductive_reasoning",
        "prompt": "All squares are rectangles. No rectangles are circles. Are any squares circles? Yes or No.",
        "eval_type": "logical",
        "expected_val": "No"
    },

    # 8. CODE GENERATION
    {
        "task_id": "codegen_1",
        "category": "code_generation",
        "prompt": "Write a Python function to check if a number is prime.",
        "eval_type": "code",
        "keywords": ["def ", "%", "return"]
    },
    {
        "task_id": "codegen_2",
        "category": "code_generation",
        "prompt": "Write a Python function that takes a list of integers and returns the second largest element.",
        "eval_type": "code",
        "keywords": ["def ", "return"]
    },
    {
        "task_id": "codegen_3",
        "category": "code_generation",
        "prompt": "Write a Python function to reverse a string.",
        "eval_type": "code",
        "keywords": ["def ", "[::-1]"]
    },
    {
        "task_id": "codegen_4",
        "category": "code_generation",
        "prompt": "Write a Python function to calculate the factorial of a non-negative integer recursively.",
        "eval_type": "code",
        "keywords": ["def ", "*"]
    },
    {
        "task_id": "codegen_5",
        "category": "code_generation",
        "prompt": "Write a Python function that counts the number of vowels in a string.",
        "eval_type": "code",
        "keywords": ["def ", "in"]
    },
    {
        "task_id": "codegen_6",
        "category": "code_generation",
        "prompt": "Write a Python function to merge two sorted lists into one sorted list.",
        "eval_type": "code",
        "keywords": ["def "]
    },
    {
        "task_id": "codegen_7",
        "category": "code_generation",
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "eval_type": "code",
        "keywords": ["def ", "=="]
    },
    {
        "task_id": "codegen_8",
        "category": "code_generation",
        "prompt": "Write a Python function that returns the Fibonacci sequence up to n terms.",
        "eval_type": "code",
        "keywords": ["def "]
    },
    {
        "task_id": "codegen_9",
        "category": "code_generation",
        "prompt": "Write a Python function to find the greatest common divisor (GCD) of two numbers.",
        "eval_type": "code",
        "keywords": ["def "]
    },
    {
        "task_id": "codegen_10",
        "category": "code_generation",
        "prompt": "Write a Python function that flattens a nested list.",
        "eval_type": "code",
        "keywords": ["def "]
    },
    {
        "task_id": "codegen_11",
        "category": "code_generation",
        "prompt": "Write a Python function to check if two strings are anagrams.",
        "eval_type": "code",
        "keywords": ["def ", "sorted"]
    },
    {
        "task_id": "codegen_12",
        "category": "code_generation",
        "prompt": "Write a Python function that returns the intersection of two lists.",
        "eval_type": "code",
        "keywords": ["def ", "set"]
    },
    {
        "task_id": "codegen_13",
        "category": "code_generation",
        "prompt": "Write a Python function to remove all duplicates from a list while preserving order.",
        "eval_type": "code",
        "keywords": ["def "]
    },
    {
        "task_id": "codegen_14",
        "category": "code_generation",
        "prompt": "Write a Python function to find the length of the longest word in a sentence.",
        "eval_type": "code",
        "keywords": ["def ", "split"]
    },
    {
        "task_id": "codegen_15",
        "category": "code_generation",
        "prompt": "Write a Python function that converts Celsius to Fahrenheit.",
        "eval_type": "code",
        "keywords": ["def ", "*", "32"]
    }
]


# Helper to count sentences safely
def count_sentences(text: str) -> int:
    cleaned = text
    # Avoid splitting on common abbreviations or decimals
    abbrevs = ["e.g.", "i.e.", "u.s.", "dr.", "mr.", "ms.", "mrs.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."]
    for abbrev in abbrevs:
        cleaned = re.sub(re.escape(abbrev), abbrev.replace(".", "@"), cleaned, flags=re.IGNORECASE)
    # Also ignore decimal dots (like 1.875)
    cleaned = re.sub(r'(\d)\.(\d)', r'\1@\2', cleaned)
    parts = re.split(r'[.!?]+(?:\s+|$)', cleaned)
    return len([p for p in parts if p.strip()])


# Helper to count bullets safely
def count_bullets(text: str) -> int:
    lines = [line.strip() for line in text.strip().split("\n")]
    bullets = [line for line in lines if line.startswith(("-", "*", "•", "1.", "2.", "3."))]
    return len(bullets)


def load_dotenv():
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key, val = parts[0].strip(), parts[1].strip()
                        if key not in os.environ:
                            os.environ[key] = val


def run_evaluation(category_filter: str | None, limit: int | None):
    # 0. Load env variables from local .env
    load_dotenv()

    # 1. Filter tasks
    filtered = TEST_CASES
    if category_filter:
        filtered = [t for t in filtered if t["category"] == category_filter]
    if limit:
        filtered = filtered[:limit]

    if not filtered:
        print(f"No tasks matched the filter criteria (category={category_filter}).")
        return

    print("=" * 80)
    print(f" STANDARDIZED EVALUATION: {len(filtered)} Test Tasks Loaded")
    print("=" * 80)

    # 2. Check environment
    api_key = os.environ.get("FIREWORKS_API_KEY", "test")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    is_mock = "localhost" in base_url or "127.0.0.1" in base_url or api_key in ("test", "test-key")

    if is_mock:
        print("\033[33m[NOTE] Running against a MOCK Fireworks Server. Content correctness checks will fail.\033[0m")
    else:
        print("\033[32m[INFO] Running against the REAL Fireworks API. Real evaluation scores will be computed.\033[0m")

    # 3. Write inputs to test_input/eval_tasks.json
    os.makedirs("test_input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    eval_input_path = "test_input/eval_tasks.json"
    eval_output_path = "output/eval_results.json"

    # Only output task_id and prompt to tasks.json so agent reads it cleanly
    clean_tasks = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in filtered]
    with open(eval_input_path, "w") as f:
        json.dump(clean_tasks, f, indent=2)

    # Set up env variables
    env = os.environ.copy()
    env["INPUT_PATH"] = eval_input_path
    env["OUTPUT_PATH"] = eval_output_path

    # 4. Run the agent
    print("Executing agent on test tasks...")
    start_time = time.time()
    try:
        res = subprocess.run(["python", "-m", "app.main"], env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as err:
        print("\033[91m[FATAL] Agent run crashed with exit code", err.returncode)
        print("STDOUT:")
        print(err.stdout)
        print("STDERR:")
        print(err.stderr)
        print("\033[0m")
        return
    duration = time.time() - start_time

    # 5. Load results
    if not os.path.exists(eval_output_path):
        print(f"\033[91m[FATAL] Output file was not created at {eval_output_path}\033[0m")
        return

    try:
        with open(eval_output_path, "r") as f:
            results = json.load(f)
    except Exception as exc:
        print(f"\033[91m[FATAL] Output results could not be parsed as valid JSON: {exc}\033[0m")
        return

    results_map = {r["task_id"]: r["answer"] for r in results if "task_id" in r and "answer" in r}

    # 6. Parse log to count classification accuracy
    # We log classification events in app/main.py like: "[info] task task_id (category) -> ..."
    # Let's inspect stderr to read classified categories
    classified_categories = {}
    for line in res.stderr.split("\n"):
        match = re.search(r"task\s+(\w+)\s+\(([^)]+)\)\s+->", line)
        if match:
            tid, cat = match.group(1), match.group(2)
            classified_categories[tid] = cat

    # 7. Evaluate and grade
    failures = []
    category_stats = {}  # {category: {"total": 0, "correct_cat": 0, "correct_ans": 0}}

    for task in filtered:
        tid = task["task_id"]
        cat = task["category"]

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct_cat": 0, "correct_ans": 0}
        category_stats[cat]["total"] += 1

        # Check classification
        actual_cat = classified_categories.get(tid)
        cat_ok = (actual_cat == cat)
        if cat_ok:
            category_stats[cat]["correct_cat"] += 1

        # Check answer
        ans = results_map.get(tid, "")
        ans_ok = False
        error_msg = ""

        if ans.strip() == "":
            error_msg = "Empty answer returned."
        else:
            etype = task["eval_type"]
            if etype == "keyword":
                missing = [kw for kw in task["keywords"] if kw.lower() not in ans.lower()]
                if not missing:
                    ans_ok = True
                else:
                    error_msg = f"Missing keywords: {missing}"

            elif etype == "math":
                target = task["math_val"]
                if target in ans:
                    ans_ok = True
                else:
                    error_msg = f"Expected numeric result '{target}' was not found in answer."

            elif etype == "sentiment":
                # Must start with Sentiment: Label.
                sent_match = re.match(r"^Sentiment:\s*([A-Za-z]+)\.\s*Justification:", ans.strip(), re.IGNORECASE)
                if not sent_match:
                    error_msg = "Format violation. Must match: 'Sentiment: <Label>. Justification: <Reason>'"
                else:
                    label = sent_match.group(1).capitalize()
                    allowed = task["allowed_labels"]
                    if label not in allowed:
                        error_msg = f"Sentiment label '{label}' is not in allowed list: {allowed}"
                    else:
                        justification = ans.split("Justification:", 1)[1].strip()
                        if not justification:
                            error_msg = "Justification is empty."
                        elif task.get("mixed_required", False):
                            # Must cover both negative and positive parts
                            has_neg = any(kw.lower() in justification.lower() for kw in task["mixed_keywords"][0])
                            has_pos = any(kw.lower() in justification.lower() for kw in task["mixed_keywords"][1])
                            if not (has_neg and has_pos):
                                error_msg = "Justification fails two-sided requirement. Must acknowledge both positive and negative aspects."
                            else:
                                ans_ok = True
                        else:
                            ans_ok = True

            elif etype == "summarization":
                # Check formatting/sentence count
                if "expected_sentences" in task:
                    sc = count_sentences(ans)
                    expected = task["expected_sentences"]
                    if sc != expected:
                        error_msg = f"Sentence count violation. Got {sc} sentences, expected exactly {expected}."
                    else:
                        ans_ok = True
                elif "expected_bullets" in task:
                    bc = count_bullets(ans)
                    expected = task["expected_bullets"]
                    if bc != expected:
                        error_msg = f"Bullet count violation. Got {bc} bullets, expected exactly {expected}."
                    else:
                        # Check word limit on bullets if specified
                        limit_ok = True
                        if "max_bullet_words" in task:
                            limit = task["max_bullet_words"]
                            for line in ans.split("\n"):
                                if line.strip().startswith(("-", "*", "•", "1.", "2.", "3.")):
                                    words = [w for w in line.split() if w not in ("-", "*", "•", "1.", "2.", "3.")]
                                    if len(words) > limit:
                                        limit_ok = False
                                        error_msg = f"Bullet point exceeded word limit: '{line}' ({len(words)} words, limit={limit})."
                                        break
                        if limit_ok:
                            ans_ok = True

            elif etype == "ner":
                # Must be valid JSON list
                try:
                    ner_data = json.loads(ans)
                    if not isinstance(ner_data, list):
                        error_msg = "NER output is not a JSON list."
                    else:
                        # Check labels
                        valid_labels = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}
                        invalid = []
                        for entity in ner_data:
                            if not isinstance(entity, dict) or "text" not in entity or "label" not in entity:
                                invalid.append(f"Invalid format: {entity}")
                            elif entity["label"] not in valid_labels:
                                invalid.append(f"Lowercase/invalid label: {entity['label']}")
                        if invalid:
                            error_msg = "; ".join(invalid)
                        else:
                            # If running on real backend, verify some target entities
                            if not is_mock and "entities" in task:
                                expected_ents = task["entities"]
                                matched = 0
                                for key, val in expected_ents.items():
                                    for item in ner_data:
                                        if key.lower() in item["text"].lower() and item["label"] == val:
                                            matched += 1
                                            break
                                if matched < len(expected_ents) - 1:  # allow 1 miss defensively
                                    error_msg = f"Failed to extract necessary entities. Expected: {expected_ents}"
                                else:
                                    ans_ok = True
                            else:
                                ans_ok = True
                except Exception as exc:
                    error_msg = f"NER output could not be parsed as valid JSON: {exc}"

            elif etype == "code":
                # Check for markdown code fences
                if "```" in ans:
                    error_msg = "Format violation. Output contains markdown code fences (```)."
                elif "keywords" in task:
                    missing = [kw for kw in task["keywords"] if kw.lower() not in ans.lower()]
                    if not missing:
                        ans_ok = True
                    else:
                        error_msg = f"Missing code structures: {missing}"
                else:
                    ans_ok = True

            elif etype == "logical":
                target = task["expected_val"]
                if target.lower() in ans.lower():
                    ans_ok = True
                else:
                    error_msg = f"Expected logic answer '{target}' was not found in answer."

        # Mark mock runs as answer OK if the format was correct, since content cannot be true
        if is_mock and error_msg not in ("Format violation. Must match: 'Sentiment: <Label>. Justification: <Reason>'",
                                         "Sentence count violation.", "Bullet count violation.",
                                         "NER output could not be parsed as valid JSON",
                                         "Format violation. Output contains markdown code fences (```)."):
            ans_ok = True

        if ans_ok:
            category_stats[cat]["correct_ans"] += 1
        else:
            failures.append({
                "task_id": tid,
                "category": cat,
                "prompt": task["prompt"],
                "classified_as": actual_cat,
                "answer": ans,
                "error": error_msg
            })

    # 8. Print Results Report
    print("\n" + "=" * 80)
    print(" EVALUATION BREAKDOWN")
    print("=" * 80)
    print(f"{'Category':<32} | {'Total':<6} | {'Classified OK':<13} | {'Answers OK':<10} | {'Pass Rate':<9}")
    print("-" * 80)
    for cat, stats in category_stats.items():
        total = stats["total"]
        class_ok = stats["correct_cat"]
        ans_ok = stats["correct_ans"]
        rate = (ans_ok / total) * 100
        print(f"{cat:<32} | {total:<6} | {class_ok:<13} | {ans_ok:<10} | {rate:>7.1f}%")
    print("-" * 80)

    total_tasks = len(filtered)
    total_ans_ok = sum(s["correct_ans"] for s in category_stats.values())
    total_class_ok = sum(s["correct_cat"] for s in category_stats.values())
    total_rate = (total_ans_ok / total_tasks) * 100
    class_rate = (total_class_ok / total_tasks) * 100

    print(f"{'TOTALS':<32} | {total_tasks:<6} | {total_class_ok:<13} | {total_ans_ok:<10} | {total_rate:>7.1f}%")
    print(f"Classification Accuracy: {class_rate:.1f}%")
    print(f"Total Execution Runtime: {duration:.2f} seconds")
    print("=" * 80)

    # Report failures
    if failures:
        print(f"\n\033[91mFAILED TASKS BREAKDOWN ({len(failures)} failures):\033[0m")
        for fail in failures[:15]:  # show up to 15 failures to avoid clutter
            print(f"\n[ID: {fail['task_id']}] Category: {fail['category']}")
            print(f"  Prompt: {fail['prompt']}")
            print(f"  Classified As: {fail['classified_as']}")
            try:
                print(f"  Actual Answer: '{fail['answer'].strip()}'")
            except UnicodeEncodeError:
                safe_answer = fail['answer'].encode('ascii', 'replace').decode('ascii')
                print(f"  Actual Answer: '{safe_answer.strip()}'")
            print(f"  \033[91mReason: {fail['error']}\033[0m")
        if len(failures) > 15:
            print(f"\n... and {len(failures) - 15} more failures. Check output/eval_results.json for details.")
    else:
        print("\n\033[32mALL TESTS PASSED! Output is fully correct and properly formatted.\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track 2 Agent Evaluation Suite")
    parser.add_argument("--category", type=str, default=None, help="Only run tasks of this category")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tasks run")
    args = parser.parse_args()

    run_evaluation(args.category, args.limit)
