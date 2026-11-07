# Indian Sign Language (ISL) Vocabulary & Reference Guide

This document defines the 10 predefined signs supported by the SignBridge hackathon prototype. These signs were selected to demonstrate a practical daily communication aid between an ISL signer and a non-signer.

---

## Supported Vocabulary

| # | Sign Name | ID | Expected Hands | Meaning & Context | Signing Pose Description |
|---|---|---|---|---|---|
| 1 | **Hello** | `HELLO` | 1 Hand | Greeting | Open hand raised near temple/forehead with palm facing outward, moving slightly outward in a polite salute/wave. |
| 2 | **Namaste** | `NAMASTE` | 2 Hands | Respectful Greeting | Both palms pressed flat together in front of the chest (Anjali Mudra), representing traditional Indian reverence. |
| 3 | **Thank You** | `THANK_YOU` | 1 Hand | Expressing gratitude | Flat hand fingers gently touching the chin or lower lip, then extending outward toward the listener with a slight nod. |
| 4 | **Yes** | `YES` | 1 Hand | Affirmation / Agreement | Closed fist held in front, nodding up and down from the wrist, or a firm thumbs-up gesture indicating confirmation. |
| 5 | **No** | `NO` | 1 Hand | Negation / Disagreement | Index finger extended upright and moving side-to-side in negation, or index and middle finger closing to meet the thumb. |
| 6 | **Please** | `PLEASE` | 1 Hand | Polite request | Open flat palm gently placed or rubbed in a gentle circular motion over the chest/heart. |
| 7 | **Help** | `HELP` | 2 Hands | Requesting assistance | Dominant hand forms a closed fist with thumb upright, placed on top of the open, upward-facing palm of the non-dominant hand. |
| 8 | **Good** | `GOOD` | 1 Hand | Approval / Quality | Clear thumbs-up gesture held firmly at chest level with fingers curled into the palm. |
| 9 | **Water** | `WATER` | 1 Hand | Need for water / drink | 'W' handshape (index, middle, and ring fingers extended upright, thumb holding pinky) tapped gently near the mouth. |
| 10 | **Goodbye** | `BYE` | 1 Hand | Parting farewell | Open palm facing the viewer, waving fingers side-to-side in a friendly parting motion. |

---

## Linguistic & Prototype Notes

1. **Sign Language vs. Spoken Grammar**:
   Indian Sign Language (ISL) has its own rich, independent grammar (e.g. Subject-Object-Verb word order, facial expressions, spatial inflections) that does not match English syntax word-for-word.
2. **Prototype Scope**:
   SignBridge acts as a limited-vocabulary communication aid. While it recognizes isolated sign glosses and assembles them into readable text and speech, it does not claim to generate full grammatical translations of complex conversational ISL.
3. **Tracking Invariance**:
   All 21 landmark points are translated relative to the wrist and scaled by the hand's span. This ensures consistent recognition regardless of hand size or distance from the camera.
