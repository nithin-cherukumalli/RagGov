# Phase 2 Human-Audit Worklist

Source: `evals/govrag_calib/staging/raw/heldout_real_v1.jsonl`.

This is a human audit sheet. S2 classifications are provisional text-only reads, not accepted labels.

Summary:
- CONTRADICTED rows reviewed: 25
- CONTRADICTED provisional counts: {'actually fine/mislabeled': 24, 'actually unsupported': 1}
- CLEAN rows spot-checked: 10
- CLEAN provisional counts: {'faithful': 5, 'needs human check': 5}
## CONTRADICTED_CLAIM Human Audit Worklist

### C01. source_id=13392

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer appears supported by Farkle Ten instructions; no explicit not-X conflict found.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how play farkle

**Answer**

> To play Farkle, players take turns rolling six dice. The goal is to have the highest score after 10 rounds. Players can collect points during their turn, but if they roll a Farkle (a roll without any points), they lose all the points they've collected during that turn and play passes to the next player. Players can choose to stop rolling at any point during their turn to avoid the risk of rolling a Farkle. The points are then recorded. Points are awarded based on the number of dice rolled, with two sets of three of a kind yielding 2,500 points, four of a kind yielding 1,000 points, and five of a kind yielding 2,000 points.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-13392-p0]
Welcome to Farkle Ten! Farkle Ten is a unique twist on the game of Farkle geared towards fast and fun online play. Choose How To Play Farkle for a simple set of instructions on how to play or browse the rest of the Farkle Ten guide for hints and tips! Object: The player with the highest score in 10 rounds of play wins! This is a modified version of Farkle geared towards faster online play allowing you to finish up a game in one series instead of waiting on the other player over and over again. How to Play: Each player takes turns rolling the dice. When it's your turn you'll roll all 6 dice.

[c1 | doc=ragtruth-13392-p1]
A Farkle occurs when the dice are rolled and no point dice appear. At this point the player loses all the point dice he/she/it has collected during that turn, and the play passes to the player to the left. No points are recorded on the scratch paper. If a player decides not to risk rolling a Farkle then he/she/it can stop rolling and the play passes to the player to the left. Any points collected during that turn are then recorded on the scratch paper. At this point the player may eat some Tostitos® Chips and salsa.

[c2 | doc=ragtruth-13392-p2]
1 Two sets of three of a kind: 2,500 points. 2  Four of a kind is scored as 1,000 points, or double the three of a kind value, or quadruple the three of a kind value.  Five of a kind is scored as 2,000 points, or triple the three of a kind value, or six times the three of a kind value.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C02. source_id=15893

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer steps are present across chimney-cap passages; no explicit contradiction found.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to make a chimney cap

**Answer**

> Based on the given passages, here's a brief guide on how to make a chimney cap:

1. Measure the top of the chimney and the length, width, and height of the flue.
2. Build the bottom of the mold out of ¾-inch plywood, using a table saw or circular saw to cut it to the appropriate size.
3. Create a cutout for the flue.
4. Place the chimney cap over the crown of the chimney, keeping it fairly centered over the top flue.
5. Bend the flange around the bottom of the chimney cap so that it conforms to the chimney crown.
6. Draw a line around the perimeter of the flange.
7. Use a 3/16 masonry bit to drill holes for the Tapcon masonry anchors, if necessary.
8. Install the chimney cap using Tapcon screws that are ¼ diameter and 1 ½ or 1 ¾ long.

Note: The passages do not provide detailed instructions on how to build the mold or the cap itself, nor do they mention any specific materials needed for the construction of the cap. Therefore, additional research or professional guidance may be required to complete the project.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15893-p0]
Place the cap over the crown of the chimney, keeping it fairly centered over the top flue. 1  Since this type of cap essentially cages in the chimney, you do not need to worry about a perfect fit. 2  Note that the crown of the chimney is the concrete area around the flues.ips. 1  Note the advantages that come with capping your chimney. 2  Chimney caps prevent excess rain, snow, and wind from coming into your chimney. 3  Caps can also keep out birds and other animals. 4  Invest in a stainless steel cap.

[c1 | doc=ragtruth-15893-p1]
Instructions. Measure the top of the chimney and the length, width and height of the flue. Build the bottom of the mold out of ¾-inch plywood. Take the measurements of the top of the chimney and add 2 inches so the cap will overhang the chimney by 2 inches on all sides.Cut the plywood to that measurement by using a table saw or circular saw. Create the cutout for the flue.ake the measurements of the top of the chimney and add 2 inches so the cap will overhang the chimney by 2 inches on all sides. Cut the plywood to that measurement by using a table saw or circular saw. Create the cutout for the flue.

[c2 | doc=ragtruth-15893-p2]
To use the adhesive to install a top mount chimney cap, place the chimney cap where you want it to be on your chimney crown (the concrete area surrounding the flue or flues) and bend the flange around the bottom of the chimney cap so that it conforms to the chimney crown. Draw a line around the perimeter of the flange.o install a top mount chimney cap with Tapcon masonry anchors, you will need to buy a 3/16 masonry bit and enough Tapcon screws to mount your chimney cap if the manufacturer has not provided them with your chimney cap. Choose screws that are ¼diameter and 1 ½ or 1 ¾ long.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C03. source_id=13701

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer follows rib-grilling/smoking procedure in context; no explicit contradiction found.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how do you make ribs on a grill

**Answer**

> Based on the provided passages, here's how to make ribs on a grill:

1. Skin and rub the ribs with a dry rub mixture (Passage 1, Step 1).
2. Preheat a grill to medium heat (Passage 2, Step 1).
3. Place the ribs on the grill, meat-side up, and brush with some of the sauce (Passage 2, Step 2).
4. Close the grill lid and cook for 5 minutes, then flip and brush the other side of the ribs (Passage 2, Step 3).
5. Continue to cook the ribs, flipping and brushing every 5 minutes, until the sauce has thickened and the ribs are heated through and have grill marks (Passage 2, Steps 4-6). This will take approximately 20 minutes for a total grilling time of 45-50 minutes.
6. Once the ribs are cooked, coat them with sauce on both sides and grill until brown (Passage 3, Step 2).
7. Let the ribs stand for 5 minutes before cutting and serving (Passage 3, Step 4).

Unable to answer based on given passages. The passages do not provide detailed instructions on how long to cook the ribs or what temperature to use for the grill.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-13701-p0]
Once you learn this proven technique for succulent smoked ribs, you’ll be surprised how easy it is to make perfect ribs, every time. 1  1. Skin and rub the ribs. 2  2. Prepping your cooker for smoking. 3  3. Smoking the ribs.  4. Glaze or sauce the 1  ribs. 5. Testing for doneness.  6. Rest and cut.

[c1 | doc=ragtruth-13701-p1]
To grill the ribs: 1Heat a gas or charcoal grill to medium (350°F to 450°F). 2Remove the ribs from the foil, place them meaty-side up on the grill, and brush them with some of the sauce. Cover the grill and cook for 5 minutes. Flip and brush the ribs and cook another 5 minutes. Continue to cook, flipping and brushing every 5 minutes, until the sauce has thickened and the ribs are heated through and have grill marks, for a total grilling time of 20 minutes.

[c2 | doc=ragtruth-13701-p2]
Grill, both sides for approx. 4 minutes per side, to seal. Then decrease temp to medium and grill for until internal temp of 150. Flipping about every 5-6 minutes. Total cooking time 25 minutes for me. Then coat with sauce on both sides and grill until brown. 2-4 more minutes. Let stand for 5 minutes before cutting.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C04. source_id=13368

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer paraphrases Bible/words passage; no explicit contradiction found.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> what the bible say about how words can destroy

**Answer**

> The Bible emphasizes the power of words, stating that they can do much more than just convey information. According to the Bible, words have the ability to destroy a person's spirit and even incite hatred and violence (passage 1). This power is not to be used lightly or maliciously, as shown in Proverbs 15:4 and Romans 12:17-18 where the use of gentle words and avoidance of evil are encouraged (passage 2). Therefore, the Bible warns about the destructive potential of words and encourages their respectful use.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-13368-p0]
Question: What does the Bible say about the power of words?. Answer: Words are not simply sounds caused by air passing through our larynx. Words have real power. God spoke the world into being by the power of His words (Hebrews 11:3), and we are in His image in part because of the power we have with words. Words do more than convey information. The power of our words can actually destroy one’s spirit, even stir up hatred and violence.

[c1 | doc=ragtruth-13368-p1]
Proverbs 15:4 ESV / 101 helpful votes. A gentle tongue is a tree of life, but perverseness in it breaks the spirit. Psalm 19:14 ESV / 95 helpful votes. Let the words of my mouth and the meditation of my heart be acceptable in your sight, O Lord, my rock and my redeemer. Romans 12:17-18 ESV / 85 helpful votes. Repay no one evil for evil, but give thought to do what is honorable in the sight of all.

[c2 | doc=ragtruth-13368-p2]
Respectfully Destroying a Bible. 1  1. Show respect to the Bible. The Bible itself doesn't contain any special instructions for its own disposal. 2  2. Bury the Bible. One way to dispose of an old Bible is to return it to the Earth with a respectful burial. 3  3. Cremate the Bible.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C05. source_id=15700

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Blood-flow sequence matches passage; no explicit contradiction found.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how the blood travels through the heart

**Answer**

> Based on the given passages, here is how the blood travels through the heart:

1. The blood enters the right atrium through the superior and inferior vena cavae and the coronary sinus.
2. From the right atrium, the blood passes through the tricuspid valve to the right ventricle.
3. From the right ventricle, the blood goes through the pulmonary semilunar valves to the pulmonary trunk.
4. From the pulmonary trunk, the blood moves into the right and left pulmonary arteries to the lungs.
5. Oxygenated blood is returned to the heart through the pulmonary veins.

Therefore, the blood travels from the right atrium to the right ventricle, then to the lungs, and back to the left atrium and ventricle, completing one loop of the circulatory system.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15700-p0]
Pathway of blood through the heart. 1. Blood enters the right atrium from the superior and inferior venae cavae, and the coronary sinus. 2. From right atrium, it goes through the tricuspid valve to the right ventricle. 3. From the right ventricle, it goes through the pulmonary semilunar valves to. the pulmonary trunk. 4. From the pulmonary trunk it moves into the right and left pulmonary. arteries to the lungs. 5. From the lungs, oxygenated blood is returned to the heart through the. pulmonary veins.

[c1 | doc=ragtruth-15700-p1]
Blood enters the heart through two large veins, the inferior and superior vena cava, emptying oxygen-poor blood from the body into the right atrium of the heart. As the atrium contracts, blood flows from your right atrium into your right ventricle through the open tricuspid valve.

[c2 | doc=ragtruth-15700-p2]
Student will learn the circulatory system's role and purpose. Student will examine anatomy of the heart and the blood vessels. Student will recognize differences and function of arteries versus veins. Student will learn the circuit and travel of blood throughout the circulatory system.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C06. source_id=15043

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Prune preparation answer mirrors passage text.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to eat prunes for constipation

**Answer**

> In a small saucepan, place prunes, water, and orange juice. Bring to boil, cover, and simmer for 30 minutes. Prunes will absorb most of the water, and the remaining water will thicken into a consistency of syrup. If there is a lot of liquid remaining, simmer, uncovered, for a few minutes until sauce reduces. Transfer hot prunes and syrup to a bowl. Pour cold milk over the prunes to cover.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15043-p0]
1 In a small saucepan, place prunes, water, and orange juice. 2  Bring to boil, cover, and simmer for 30 minutes. 3  Prunes will absorb most of the water, and the remaining water will thicken into a consistency of syrup. 4  If there is a lot of liquid remaining, simmer, uncovered, for a few minutes until sauce reduces.

[c1 | doc=ragtruth-15043-p1]
Directions. 1  In a small saucepan, place prunes, water, and orange juice. 2  Bring to boil, cover, and simmer for 30 minutes. 3  Prunes will absorb most of the water, and the remaining water will thicken into a consistency of syrup. 4  If there is a lot of liquid remaining, simmer, uncovered, for a few minutes until sauce reduces.

[c2 | doc=ragtruth-15043-p2]
1 Prunes will absorb most of the water, and the remaining water will thicken into a consistency of syrup. 2  If there is a lot of liquid remaining, simmer, uncovered, for a few minutes until sauce reduces. 3  Transfer hot prunes and syrup to a bowl. 4  Pour cold milk over the prunes to cover.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C07. source_id=13084

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Parkinson symptom list appears supported by retrieved text.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> what are the symptoms of parkinson's

**Answer**

> Based on the provided passages, the symptoms of Parkinson's disease include:

1. Tremors or shaking, typically starting on one side of the body
2. Stiffness and aching muscles
3. Reduced arm swing on one side while walking
4. Rigidity affecting the muscles of the legs, face, neck, or other parts of the body.

Please note that these symptoms may vary from person to person, and not everyone will experience all of them. Some people may experience non-motor symptoms such as pain and depression as well.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-13084-p0]
The symptoms most often associated with Parkinson's affect movement. These are often called motor symptoms. Parkinson's is a very individual condition, with each person experiencing different symptoms.However, there are other symptoms that aren't related to movement, such as pain and depression.These are known as non-motor symptoms. The symptoms that people with Parkinson's may get are listed below.It's important to remember that everyone's experience of Parkinson's is different. Not everyone will experience all of these symptoms.hese are known as non-motor symptoms. The symptoms that people with Parkinson's may get are listed below. It's important to remember that everyone's experience of Parkinson's is different. Not everyone will experience all of these symptoms.

[c1 | doc=ragtruth-13084-p1]
Other common symptoms. Besides tremor, the most common symptoms include: 1  Stiff muscles (rigidity) and aching muscles. 2  One of the most common early signs of Parkinson's is a reduced arm swing on one side when you walk.3  This is caused by rigid muscles. 4  Rigidity can also affect the muscles of the legs, face, neck, or other parts of the body.he most common cause of non-Parkinson's tremor is essential tremor. It's a treatable condition that is often wrongly diagnosed as Parkinson's. Besides tremor, the most common symptoms include: 1  Stiff muscles (rigidity) and aching muscles.

[c2 | doc=ragtruth-13084-p2]
Parkinson's disease symptoms and signs may vary from person to person. Early signs may be mild and may go unnoticed. Symptoms often begin on one side of your body and usually remain worse on that side, even after symptoms begin to affect both sides.Parkinson's signs and symptoms may include: 1  Tremor. 2  A tremor, or shaking, usually begins in a limb, often your hand or fingers.ymptoms often begin on one side of your body and usually remain worse on that side, even after symptoms begin to affect both sides. Parkinson's signs and symptoms may include: 1  Tremor.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C08. source_id=17239

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: iTunes/Apple Music radio steps match passage.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to listen to itunes internet radio on iphone

**Answer**

> To listen to iTunes internet radio on an iPhone, open the Music app and tap the Radio tab at the bottom of the screen. If prompted, sign in with your Apple ID.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-17239-p0]
Start listening to Apple Music Radio. To listen on your iPhone, iPad, iPod touch, open the Music app and tap the Radio tab at the bottom of the screen. You might be asked to sign in with your Apple ID. To listen on your Mac or PC, open iTunes.From the navigation bar at the top of iTunes, choose Radio.You might be asked to sign in with your Apple ID.dd songs from Apple Music Radio to your library. If you’re a member of Apple Music, you can add songs from Apple Music Radio to your library. 1  Tap the More Options icon to the right of the song, album, or playlist. 2  Tap Add to My Music.

[c1 | doc=ragtruth-17239-p1]
Add songs from Apple Music Radio to your library. If you’re a member of Apple Music, you can add songs from Apple Music Radio to your library. 1  Tap the More Options icon to the right of the song, album, or playlist. 2  Tap Add to My Music.dd songs from Apple Music Radio to your library. If you’re a member of Apple Music, you can add songs from Apple Music Radio to your library. 1  Tap the More Options icon to the right of the song, album, or playlist. 2  Tap Add to My Music.

[c2 | doc=ragtruth-17239-p2]
Tune in to Internet stations. Choose Music from the Library pop-up menu. Click Internet. To see the available stations, click the triangle next to the type of music you want to listen to.To tune in, double-click a station.une in to Internet stations. Choose Music from the Library pop-up menu. Click Internet. To see the available stations, click the triangle next to the type of music you want to listen to.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C09. source_id=13214

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Rump-roast oven instructions match passage.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to make rump roast in oven

**Answer**

> To make rump roast in an oven, follow these steps:

1. Preheat the oven to 500 degrees.
2. Make 4-5 slits in the roast and fill the holes with salt, pepper, and 1/2 of a clove of garlic. Salt and pepper the rest of the roast.
3. Sear the roast in the oven for 20 minutes.
4. In a large Dutch oven or frying pan, heat oil and place the roast inside. Pour wine and broth over the meat and cover the roast with a lid or aluminum foil. Place in the oven and roast for 1 1/2 hours, or 30 minutes per pound.
5. Test the roast to ensure it's done. Remove the lid or foil from the pan and insert a meat thermometer into the roast.
6. Once the roast is cooked, remove it from the oven and let it rest for 10-15 minutes before serving.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-13214-p0]
Directions See How It's Made. 1  DO NOT USE A GLASS PAN FOR THIS RECIPE. 2  Pre-Heat the oven to 500 degrees. 3  Prep the roast by making 4-5 slits in the roast and filling the holes with salt, pepper and 1/2 of a clove of Garlic.  Now Salt and Pepper the rest of the 1  roast. Put the Roast in the oven and sear for 20 minutes.

[c1 | doc=ragtruth-13214-p1]
Braised Rump Roast. 1  1. Trim the roast of extra fat. 2  2. Preheat the oven to 325 degrees Fahrenheit (162 degrees Celsius). 3  3. Heat oil in a large dutch oven or frying pan.  4. Place the roast in a dutch oven or roasting 1  pan. 5. Pour the wine and broth over the meat.  6. Cover the roast with a lid or aluminum foil. Place in the oven and roast for 1 1/2 hours, or 30 minutes per pound. 7. Test the roast to make sure it's done. Remove the lid or foil from the pan and insert a meat thermometer into the roast.

[c2 | doc=ragtruth-13214-p2]
Place the roast in the pan and sear it on all sides until golden-brown, about five minutes total. Use tongs to turn the rump roast over then needed. Heat the oven to 300 degrees. Remove the rump roast and set it aside. Turn the heat on the stove to medium and add aromatic ingredients to the Dutch oven or pan. Use 1/2 cup of standard mirepoix -- 2 parts diced onion to 1 part each diced carrots and celery -- per pound of roast.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C10. source_id=15941

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Strut/shock distinction is supported by passage.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> difference between struts and shocks

**Answer**

> Based on the given passages, the main difference between struts and shocks is that struts provide structural support for the vehicle and combine shock absorption with steering elements, while shocks only resist the bouncing using hydraulics and do not provide any structural support. Additionally, struts are typically used in car suspension systems, while shocks are more common in trucks, SUVs, and vans.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15941-p0]
On Automobiles we loosely use the word 'damper' in place of 'shock'. A Strut is essentially a shock absorber and a spring fitted into one unit. On some cars, the damper (or technically the shock absorber) is seperate from the spring. If the Spring and shock are mounted together as a single unit, it is called a strut. Now to conclude, a damper is any material/device in general that can absorb vibration. A shock absorber is a type of damper known as friction damper. A Strut is a shock (damper) with a spring as one unit. Dampers and shock-absorbers are the same. Both of them perform the same function- to quickly phase out the vibrations from the spring that has absorbed the shocks on road in the form of vibrations.

[c1 | doc=ragtruth-15941-p1]
A shock absorber resists the bouncing using hydraulics. That’s all it does. Struts are more complex. They combine shock absorption with steering elements and support for the vehicle. Shocks are more common on trucks and SUVs, while you usually find struts on cars. But sometimes you find both—one in the front and the other in the back. Struts are more expensive, because they are more complex.

[c2 | doc=ragtruth-15941-p2]
So, what exactly is the difference between shocks and struts? Shocks and struts are two different suspension parts that perform similar functions on various suspension systems. In strut suspension systems, the spring is held by the strut, and the strut provides structural support for the vehicle. Struts are more common on cars than on trucks and larger vehicles. Shocks, on the other hand, provide no structural support for your vehicle. They are mounted directly to the vehicle’s frame at one end and the lower suspension on the other end. Shocks are most commonly found on trucks, SUVs and vans.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C11. source_id=16676

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer abstains; source contains related advice, but there is no contradiction.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to save money to leave my marriage

**Answer**

> Unable to answer based on given passages.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-16676-p0]
Don’t think of yourself as “trapped” – though I know that if you’re looking for ways to get money to leave your husband, that’s exactly how you feel. Focus on the fact that you WILL get money to leave your husband. It’s just a matter of time and planning. And taking action.

[c1 | doc=ragtruth-16676-p1]
When a person comes to me asking, can one spouse save a marriage, my answer is a qualified yes.. I see one spouse saving a marriage almost every week. The way it's done is that the one spouse teaches the other spouse how to negotiate fairly.

[c2 | doc=ragtruth-16676-p2]
May you see that money is helpful, but it won’t make you happy. I pray for wisdom and guidance as you gain insight into yourself and your situation, and for clarity as you think about how to get money to leave your husband. I pray for peace and joy, and a strong connection with your Creator. Amen.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C12. source_id=15699

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Blood-flow sequence matches passage; no explicit contradiction found.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how the blood travels through the heart

**Answer**

> Based on the provided passages, here is how the blood travels through the heart:

1. Passage 1: The blood enters the right atrium from the superior and inferior venae cavae, and the coronary sinus. (Step 1)
2. From the right atrium, it goes through the tricuspid valve to the right ventricle. (Step 2)
3. From the right ventricle, it goes through the pulmonary semilunar valves to the pulmonary trunk. (Step 3)
4. From the pulmonary trunk, it moves into the right and left pulmonary arteries to the lungs. (Step 4)
5. From the lungs, oxygenated blood is returned to the heart through the pulmonary veins. (Step 5)

Therefore, the blood travels through the heart in the following pathway: Right Atrium > Tricuspid Valve > Right Ventricle > Pulmonary Semilunar Valves > Pulmonary Trunk > Right and Left Pulmonary Arteries > Lungs > Pulmonary Veins > Right Atrium.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15699-p0]
Pathway of blood through the heart. 1. Blood enters the right atrium from the superior and inferior venae cavae, and the coronary sinus. 2. From right atrium, it goes through the tricuspid valve to the right ventricle. 3. From the right ventricle, it goes through the pulmonary semilunar valves to. the pulmonary trunk. 4. From the pulmonary trunk it moves into the right and left pulmonary. arteries to the lungs. 5. From the lungs, oxygenated blood is returned to the heart through the. pulmonary veins.

[c1 | doc=ragtruth-15699-p1]
Blood enters the heart through two large veins, the inferior and superior vena cava, emptying oxygen-poor blood from the body into the right atrium of the heart. As the atrium contracts, blood flows from your right atrium into your right ventricle through the open tricuspid valve.

[c2 | doc=ragtruth-15699-p2]
Student will learn the circulatory system's role and purpose. Student will examine anatomy of the heart and the blood vessels. Student will recognize differences and function of arteries versus veins. Student will learn the circuit and travel of blood throughout the circulatory system.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C13. source_id=15821

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: iPhone charging/troubleshooting steps are supported by passage.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> iphone wont turn on from charging

**Answer**

> Based on the given passages, here's a possible solution to the problem of an iPhone that won't turn on from charging:

1. Plug the iPhone's USB cable into the dock connector, but not into your computer.
2. Press and hold the Sleep/Wake button and the Home button at the same time for at least ten seconds, until either the Apple logo or the battery symbol appears.
3. If the display turns on, go to step 4.
4. Restore the iPhone. Connect the iPhone to the iPhone's USB power adapter and let it charge for at least ten minutes.
5. After at least 15 minutes, if the home screen appears, the iPhone should be working.

Therefore, the solution is: Plug the iPhone's USB cable into the dock connector, press and hold the Sleep/Wake button and the Home button, restore the iPhone, and let it charge for at least ten minutes.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15821-p0]
If your device still won't turn on or start up. Plug in your device and let it charge for up to 1 hour. After a few minutes, you should see the charging screen. If you don't see the charging screen within an hour, or you see the connect to power screen, check your connector, USB cable, and power adapter.lug in your device and let it charge for up to 1 hour. After a few minutes, you should see the charging screen. If you don't see the charging screen within an hour, or you see the connect to power screen, check your connector, USB cable, and power adapter.

[c1 | doc=ragtruth-15821-p1]
So, your iPhone could be refusing to turn on because its battery doesn't have enough juice. In that case, plug your iPhone into a wall charger or into your computer (plug it directly into the computer; some USB ports on keyboards don't have enough power to charge iPhones).ormally, you can just sync your iPhone and click the Restore button in iTunes, but if it won't turn on, you need a different technique: 1  Plug the iPhone's USB cable into the dock connector, but not into your computer.

[c2 | doc=ragtruth-15821-p2]
1 To reset, press and hold the Sleep/Wake button and the Home button at the same time for at least ten seconds, until either the Apple logo or the battery symbol appears. 2  If the display turns on, go to step 4. 3  If the display remains black, go to next step. 4  Restore the iPhone. Connect the iPhone to the iPhone's USB power adapter and let it charge for at least ten minutes. 2  After at least 15 minutes, if: 3  The home screen appears: The iPhone should be working.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C14. source_id=17183

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Hard-boil-egg steps appear supported by cooking/video context.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to hard boil eggs youtube

**Answer**

> Based on the given passages, here's a step-by-step guide on how to hard boil eggs:

1. Place 6 cold eggs in a saucepan and fill with cold water, covering the eggs by an inch.
2. Set the pan over high heat and bring the water to a boil, uncovered.
3. Once the water reaches a rolling boil, let it slow boil for 12 minutes.
4. Remove the pan from heat and add ice to stop the cooking process.

That's it! You should now have perfectly hard-boiled eggs. Enjoy!

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-17183-p0]
How to Hard Boil Eggs perfectly every time! Visit http://foodwishes.com, to get the ingredients and more info, and watch over 600 free video recipes.Thanks and enjoy! 1  TV-Y. 2  Food Wishes Recipes. 3  3. 4  8. 5  2:21. 6  Chef John.ow to Hard Boil Eggs perfectly every time! Visit http://foodwishes.com, to get the ingredients and more info, and watch over 600 free video recipes. Thanks and enjoy! 1  TV-Y. 2  Food Wishes Recipes. 3  3. 4  8. 5  2:21.

[c1 | doc=ragtruth-17183-p1]
Instructions. 1  Put the eggs in a pan of cold water: Place 6 cold eggs in a saucepan and fill with cold water, covering the eggs by an inch. 2  Bring the water to a rolling boil: Set the pan over high heat and bring the water to a boil, uncovered.3  Turn off the heat and cover the pan.nstructions. 1  Put the eggs in a pan of cold water: Place 6 cold eggs in a saucepan and fill with cold water, covering the eggs by an inch. 2  Bring the water to a rolling boil: Set the pan over high heat and bring the water to a boil, uncovered. 3  The water should come to a full, rolling boil.

[c2 | doc=ragtruth-17183-p2]
1 Make sure the eggs are covered by 2 inches of water. 2  When the water reaches a rolling boil, let them slow boil 12 minutes; remove from heat. 3  Stop the cooking process by adding ice to to the water. 4  Slightly shortening the recommended length of boiling time leaves the eggs done, but without drying the yolks. Make sure the eggs are covered by 2 inches of water. 2  When the water reaches a rolling boil, let them slow boil 12 minutes; remove from heat. 3  Stop the cooking process by adding ice to to the water. 4  Slightly shortening the recommended length of boiling time leaves the eggs done, but without drying the yolks.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C15. source_id=17329

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Tibetan Terrier pet traits appear supported by context.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> what are tibetan terriers like as pets

**Answer**

> Tibetan Terriers are intelligent, mischievous, and sensitive dogs that make excellent companions. They are generally good with other dogs and pets if they have been raised with them. Tibetan Terriers require frequent grooming and make great watchdogs, as they are alert and will bark at anything unusual. Socialization is important for these dogs to ensure they grow up to be well-rounded.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-17329-p0]
Origin. Even though it may have the word terrier in its name, the Tibetan Terrier is only a terrier in size. Bred and raised in monasteries by lamas almost 2,000 years ago, the Tibetan Terrier was known as the “Holy Dogs of Tibet.” This dog was treasured by the lamas, and was considered good luck charms.t’s pretty hard to resist this shaggy dog! The Tibetan Terrier is an intelligent and mischievous dog that will be dedicated to its family. In fact, its sensitivity to the moods of its owners makes this dog an excellent companion that can be happy in almost any home environment.

[c1 | doc=ragtruth-17329-p1]
Shelters have dogs of all breeds, including purebred Tibetan Terriers. These dogs sometimes end up in shelters because a Tibetan Terrier dog breeder, or someone who had Tibetan Terrier puppies sale was not able to find a home for all the dogs.ith such a huge overpopulation problem, looking for a Tibetan Terrier dog for sale, or Tibetan Terrier puppies breeders, sadly contributes to this tragedy. Rescue groups that specialize in adoption work very, very hard to help adult dogs find good homes. Rescues are almost always made up of a group of volunteers.

[c2 | doc=ragtruth-17329-p2]
1 Tibetan Terriers generally do well with dogs and other pets, especially if they have been raised with them. 2  The Tibetan Terrier requires frequent brushing and a bath at least once per month. 3  Tibetan Terriers make great watchdogs and will bark when they see or hear anything unusual.ibetan Terriers are known for adaptability and a sense of humor. Like every dog, Tibetan Terriers need early socialization — exposure to many different people, sights, sounds, and experiences — when they're young. Socialization helps ensure that your Tibetan Terrier puppy grows up to be a well-rounded dog.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C16. source_id=15363

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Baking-soda underarm odor guidance appears supported.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to get rid of underarm odor with baking soda

**Answer**

> Based on the provided passages, here are some ways to get rid of underarm odor using baking soda:

1. Massage a baking soda paste onto the underarms while showering or bathing, let it set for 2-3 minutes before washing the area as usual. This will help kill the bacteria that causes the body odor. (Passage 2)
2. Dust baking soda onto the underarms and wait for 20 minutes before washing it with warm water. Another method is to mix lemon and water drops with the baking soda paste and apply it to the skin for 20 minutes. (Passage 3)
3. Add baking soda to a homemade deodorant or antiperspirant recipe to help neutralize underarm odor. (Passage 1)

Unable to answer based on given passages. The passages do not provide detailed instructions on how to apply the baking soda paste or the recommended frequency of use.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15363-p0]
Sweat and underarm odor is an unfortunate fact of life that sometimes leaves behind an unpleasant smell on your clothing. Perspiration is your body's natural way to maintain a lower temperature, and it's not the sweat that's causing the odor but the bacteria found lurking under your arms.weat and underarm odor is an unfortunate fact of life that sometimes leaves behind an unpleasant smell on your clothing. Perspiration is your body's natural way to maintain a lower temperature, and it's not the sweat that's causing the odor but the bacteria found lurking under your arms.

[c1 | doc=ragtruth-15363-p1]
Baking soda helps get rid of external body odor by killing the bacteria that grows in areas like underarms, breast creases, and the inner thighs. While in the shower or bath, massage a baking soda paste onto desired areas (paste recipe below). Let set for 2 or 3 minutes, then wash the area as usual.The baking soda paste will help kill the bacteria that causes the body odor, leaving the area smelling fresher longer.ow to Make Baking Soda Paste. Mix together 3 tablespoons of baking soda and 1 tablespoon of water using a spoon. Mush the mixture around in the bowl until it forms a paste. That’s it! Note: If baking soda gets into cuts or scrapes, it burns.

[c2 | doc=ragtruth-15363-p2]
To use this therapy dust baking soda to your underarms and wait for 20 minutes before washing it with warm water. Another way to use this remedy is by adding lemon and water drops to it and then applying this paste to skin for 20 minutes. Use this remedy once daily to get rid of underarm odor quickly.o eliminate and stop armpit odor naturally use home remedies like vinegar, baking soda, lemon, tea tree oil, aloe vera and honey. Following these tips daily will eliminate the bad smell from underarms in a week. Bad odor from body can lead to a very embarrassing situation.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C17. source_id=12561

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Pulmonary embolism symptom list appears supported.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> signs symptoms of pulmonary embolism

**Answer**

> Based on the provided passages, the signs and symptoms of pulmonary embolism are as follows:

1. Cough: According to passage 1, pulmonary embolism may cause a sudden cough that produces bloody sputum.
2. Chest pain: Passages 2 and 3 mention that pulmonary embolism can cause chest pain that may extend into the arm, jaw, neck, and shoulder, and may be accompanied by rapid breathing and rapid heartbeat.
3. Shortness of breath: Passages 1 and 3 both describe shortness of breath as a symptom of pulmonary embolism, which may occur suddenly.
4. Rapid heart rate: Passages 2 and 3 both mention rapid heart rate as a symptom of pulmonary embolism.
5. Rapid breathing: Passages 2 and 3 both describe rapid breathing as a symptom of pulmonary embolism.
6. Anxiety: According to passage 2, pulmonary embolism may cause anxiety.
7. Bloody sputum: Passage 1 mentions that pulmonary embolism may produce significant amounts of bloody sputum.
8. Pink, foamy mucus: Passage 7 describes coughing up pink, foamy mucus as a symptom of pulmonary embolism.
9. Fainting: Passages 2 and 3 both mention fainting as a possible symptom of pulmonary embolism.
10. Shock: Passage 3 lists signs of shock as a possible sympto
...[truncated for audit sheet]

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-12561-p0]
Symptoms of pulmonary embolism may be vague, or they may resemble symptoms associated with other diseases. Symptoms can include: 1  Cough. 2  Begins suddenly. 3  May produce bloody sputum (significant amounts of visible blood or lightly blood streaked sputum).

[c1 | doc=ragtruth-12561-p1]
This may be gradual or sudden. Other symptoms of a pulmonary embolism include: 1  anxiety. 2  clammy or bluish skin. 3  chest pain that may extend into your arm, jaw, neck, and shoulder. 4  fainting. 5  irregular heartbeat. 6  lightheadedness. 7  rapid breathing. 8  rapid heartbeat. 9  restlessness. 10  spitting up blood. 11  weak pulse.

[c2 | doc=ragtruth-12561-p2]
The symptoms of pulmonary embolism may include: 1  Shortness of breath that may occur suddenly. 2  Sudden, sharp chest pain that may become worse with deep breathing or coughing. 3  Rapid heart rate. 4  Rapid breathing. 5  Sweating. 6  Anxiety. 7  Coughing up blood or pink, foamy mucus. 8  Fainting. 9  Heart palpitations. 10  Signs of shock.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C18. source_id=15279

- Provisional S2 read: **actually unsupported**
- Provisional rationale: Answer extrapolates airport rental workflow/purchase details beyond the passages; not an explicit contradiction.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how do you purchase a rental car at airport

**Answer**

> Sure! Here's how you can purchase a rental car at an airport based on the provided passages:

1. Look for the rental car companies operating at the airport where you arrive. You can find this information on the airport's website or by contacting the airport directly (Passage 2).
2. Choose a rental car company and proceed to their counter at the airport. You can locate the counter based on the information provided on the airport's website or by asking airport staff (Passage 3).
3. Present your reservation confirmation and identification documents to the rental car representative at the counter. They will provide you with the rental car keys and any additional documentation required (Passage 1).
4. Review the terms and conditions of the rental agreement and ensure that everything is in order before finalizing the rental process.

Unable to answer based on given passages. The passages do not provide detailed instructions on how to purchase a rental car at an airport beyond locating the rental car companies operating at the airport and presenting the reservation confirmation at the counter.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15279-p0]
Let Thrifty help your travel plans go smoothly with an airport car rental deal at one of our 120 in-terminal locations.Not only will you get great airport car rental rates, you can make the airport car rental process even smoother, faster and more rewarding when you sign up for our Blue Chip Express Rental Program.hat way, we’ll know what kind of airport car rental you prefer, which options you normally want, which credit card to use, and more. Once you receive your Blue Chip number, you can use it whenever you-or your travel agent-place a reservation.

[c1 | doc=ragtruth-15279-p1]
Tip. 1  If you are picking up your rental car at an airport, a way to select a car company is to see what is available near your arrival terminal. 2  Most airport websites list rental agencies that operate from that airport. 3  If you do not wish to rent a car over the Internet, you also rent one over the phone.ow to Rent a Rental Car. Log on to a rental car website. Step 2. Enter the necessary information in the data fields, such as pick-up city or location, date range and time frame.

[c2 | doc=ragtruth-15279-p2]
ON-TERMINAL – The rental car counter that is on airport property with transportation to the rental vehicles provided by shuttle service and the elapsed time to the rental office will not exceed 30 minutes from the time the traveler requested pick-up service.FF-TERMINAL – The rental car counter and vehicles that are not physically located on the airport property and the elapsed time to the rental office will not exceed 30 minutes from the time the traveler requested pick-up service.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C19. source_id=16495

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer abstains; source contains prostate screening decision guidance, but no contradiction.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> what yearly screenings are recommended for early detection of prostate cancer

**Answer**

> Unable to answer based on given passages.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-16495-p0]
Screening means checking your body for cancer before you have symptoms. Getting screening tests regularly may find breast, cervical, and colorectal (colon) cancers early, when treatment is likely to work best. Lung cancer screening is recommended for some people who are at high risk.

[c1 | doc=ragtruth-16495-p1]
The American Cancer Society (ACS) recommends that men have a chance to make an informed decision with their health care provider about whether to be screened for prostate cancer. The decision should be made after getting information about the uncertainties, risks, and potential benefits of prostate cancer screening.

[c2 | doc=ragtruth-16495-p2]
Prostate Cancer. Prostate cancer is the most commonly diagnosed cancer, second only to skin cancer, affecting American men today. If you are an African American man or a man with a family history of prostate cancer, the likelihood of a prostate cancer diagnosis increases significantly.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C20. source_id=15487

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer abstains despite source including hickey-giving guidance; no contradiction.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to leave hickeys

**Answer**

> Unable to answer based on given passages.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-15487-p0]
Using Home Remedies. 1  1. Apply ice to the hickey. Placing an ice pack on the hickey as soon as possible can help to constrict blood vessels and reduce swelling. 2  2. Place heat on your hickey. If your hickey is swollen after two days, you may want to apply a warm compress to the area. 3  3. Apply some aloe vera to the hickey.

[c1 | doc=ragtruth-15487-p1]
What you DON'T want to do (based on my personal experience with others): 1  Slobber. 2  Bite too hard, too deep, too fast. 3  Bite straight down into the skin.  Nip or give small, shorter 1  bites. Initiate a bite before asking permission.

[c2 | doc=ragtruth-15487-p2]
Learning how to give a proper hickey will most certainly leave behind your mark of lust. Find a private place for you and your partner to be intimate without interruption. Rid yourselves of all distractions. Turn off the television, radio, cell phone and pager. Focus your entire attention on your partner. Begin to kiss and caress each other in a playful yet seductive manner. Move from kissing your partner's lips down to the neck. Slowly kiss the neck and decide on a place where you want to leave the hickey.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C21. source_id=13480

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Roasted Brussels sprouts steps appear supported.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to roast package frozen brussel sprouts

**Answer**

> Based on the given passages, here is how you can roast package frozen Brussels sprouts:

Passage 1: Preheat oven to 400 degrees F. Cut off the brown ends of the Brussels sprouts and pull off any yellow outer leaves. Mix them in a bowl with olive oil, salt, and pepper. Pour them on a sheet pan and roast for 35 to 40 minutes, until crisp on the outside and tender on the inside.

Passage 2: Directions:

1. Take frozen Brussels sprouts and place in a large bowl.
2. Pour olive oil and kosher salt over the sprouts and mix well.
3. Place the sprouts, evenly spaced and not stacked, on a baking sheet and place in a 400-degree oven for 40-45 minutes.

Passage 3: Mix the Brussels sprouts in a bowl with olive oil, salt, and pepper. Pour them on a sheet pan and roast for 35 to 40 minutes, until crisp on the outside and tender on the inside. Shake the pan from time to time to brown the sprouts evenly. Sprinkle with more kosher salt (I like these salty like French fries), and serve immediately.

Therefore, to roast package frozen Brussels sprouts, you should preheat your oven to 400 degrees F, cut off the brown ends and remove any yellow outer leaves, mix the sprouts with olive oil, salt, and pep
...[truncated for audit sheet]

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-13480-p0]
Preheat oven to 400 degrees F. Cut off the brown ends of the Brussels sprouts and pull off any yellow outer leaves. Mix them in a bowl with the olive oil, salt and pepper. Pour them on a sheet pan and roast for 35 to 40 minutes, until crisp on the outside and tender on the inside.

[c1 | doc=ragtruth-13480-p1]
Directions. 1  Take frozen brussels sprouts and place in a large bowl. 2  Pour olive oil and kosher salt over sprouts and mix well. 3  Place sprouts, evenly spaced and not stacked, on a baking sheet and place in a 400 degree oven for 40-45 minutes.

[c2 | doc=ragtruth-13480-p2]
Mix them in a bowl with the olive oil, salt and pepper. Pour them on a sheet pan and roast for 35 to 40 minutes, until crisp on the outside and tender on the inside. Shake the pan from time to time to brown the sprouts evenly. Sprinkle with more kosher salt (I like these salty like French fries), and serve immediately.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C22. source_id=16392

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Answer abstains; source gives selective breeding contrast but no contradiction.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> what is the difference between inbreeding and selective breeding

**Answer**

> Unable to answer based on given passages.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-16392-p0]
Selective breeding is a method of breeding that allows only those individual organisms with desired characteristics to produce the next generation. Humans use selective breeding, which takes advantage of naturally occurring genetic variation, to pass desired traits on to the next generation of organisms.

[c1 | doc=ragtruth-16392-p1]
Selective breeding is a method of breeding that allows only those individual organisms with desired characteristics to produce the next generation. Humans use selective breedi … ng, which takes advantage of naturally occurring genetic variation, to pass desired traits on to the next generation of organisms.

[c2 | doc=ragtruth-16392-p2]
New varieties. Natural selection and selective breeding can both cause changes in animals and plants. The difference between the two is that natural selection happens naturally, but selective breeding only occurs when humans intervene. For this reason selective breeding is sometimes called artificial selection.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C23. source_id=11925

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Weather forecast answer mirrors passage values.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> weather forecast in long beach

**Answer**

> Based on the provided passages, here is the current weather forecast for Long Beach:

As of Monday, April 24:

* Temperature: 70 degrees Fahrenheit (°F)
* Sky Condition: Sunny
* Chance of Rain: 53%
* Wind Speed: 6 mph (South)

As of Tuesday, April 25:

* Temperature: 68 °F
* Sky Condition: Sunny
* Chance of Rain: 58%
* Wind Speed: 7 mph (South-Southwest)

Unable to answer based on given passages. The passages do not provide information beyond the current temperature, sky condition, and chance of rain for Long Beach.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-11925-p0]
Monday:The Long Beach forecast for Apr 24 is 70 degrees and Sunny. There is 53 percentage chance of rain and 6 mph winds from the South. Tuesday:The Long Beach forecast for Apr 25 is 68 degrees and Sunny. There is 58 percentage chance of rain and 7 mph winds from the South-Southwest.

[c1 | doc=ragtruth-11925-p1]
Long Beach 10 Day Weather. 1  Thursday:The Long Beach forecast for Apr 20 is 73 degrees and Sunny. There is 35 percentage chance of rain and 9 mph winds from the West. 2  Friday:The Long Beach forecast for Apr 21 is 82 degrees and Sunny. There is 21 percentage chance of rain and 3 mph winds from the Northwest.

[c2 | doc=ragtruth-11925-p2]
For more than 20 years Earth Networks has operated the world’s largest and most comprehensive weather observation, lightning detection, and climate networks. We are now leveraging our big data smarts to deliver on the promise of IoT.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C24. source_id=16200

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Leave-email guidance is supported by passage.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to write an email for leave

**Answer**

> To write an email for leave, you should start with a clear subject line such as "Application of leave for 10 days." The body of the email should be addressed respectfully, for example starting with "Sir," or "Dear [recipient's name]." In the email, specify the number of working days you wish to take off and the exact dates. Also, include the reason for your leave. For instance, if you're going on a family vacation, mention that. The format and writing style of a leave application email should be the same as a traditional leave application. It's important to remember that emails can reach your recipient instantly, so your leave could be approved quickly.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-16200-p0]
An example will be just perfect to give you a direction in applying for leave. Subject: Application of leave for 10 days. Sir, I wish to apply for 10 working days leave from the 15th of August 2013 till the end of the month (15th – 30th August 2013) as I am going on a family vacation.

[c1 | doc=ragtruth-16200-p1]
Emails leave letters reaches the intended person instantly and you can get your leave approved almost in few minutes to hours. The format and writing style are the same when you write a leave application in email too.

[c2 | doc=ragtruth-16200-p2]
Well, whether or not... An example will be just perfect to give you a direction in applying for leave. Subject: Application of leave for 10 days. Sir, I wish to apply for 10 working days leave from the 15th of August 2013 till the end of the month (15th – 30th August 2013) as I am going on a family vacation.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

### C25. source_id=17324

- Provisional S2 read: **actually fine/mislabeled**
- Provisional rationale: Chinese broccoli trimming/cooking steps are supported by passages.
- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure
- Human notes:

**Query**

> how to trim chinese broccoli

**Answer**

> To trim Chinese broccoli, follow these steps:

1. Rinse the Chinese broccoli under tap and shake off excess water.
2. Wrap it in cling wrap and microwave for 1½ minutes, checking periodically to ensure it is cooked but still firm. If needed, cook in 30-second bursts.
3. Remove the Chinese broccoli from the cling wrap immediately after cooking to prevent further steaming.
4. Stack the Chinese broccoli together and cut into 4/10cm lengths.
5. Combine water and corn flour in a small pot and mix until dissolved.
6. Add the remaining ingredients and bring to a boil on the stove. Allow to thicken for 30 seconds before removing from the stove.
7. Drizzle the sauce over the Chinese broccoli and serve.

**Cited / retrieved passage text**

```text
[c0 | doc=ragtruth-17324-p0]
1 Stack the Chinese broccoli together and cut into 4/10cm lengths and stack on top of each other. 2  Combine water and corn flour in small pot, mix to dissolve. 3  Then add remaining ingredients, bring to boil on stove, boil for 30 seconds to allow to thicken, then remove from stove. 4  Drizzle over Chinese broccoli and serve.

[c1 | doc=ragtruth-17324-p1]
1 Trim the ends of the Chinese broccoli, rinse under tap, shake off excess water than wrap in cling wrap. 2  Microwave for 1½ minutes-check if cooked but still firm. 3  If it needs more cooking, do it in 30 second bursts.

[c2 | doc=ragtruth-17324-p2]
1 Once cooked, remove from cling wrap immediately, otherwise it will keep steaming. 2  Be careful when unwrapping it as it is hot! 3  Stack the Chinese broccoli together and cut into 4/10cm lengths and stack on top of each other. 4  Combine water and corn flour in small pot, mix to dissolve.
```

**Human prompt**

> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.

## CLEAN Spot-Check Worklist

### K01. source_id=5ab345db55429969a97a8122

- Source dataset: `hotpotqa`
- Provisional S2 read: **faithful**
- Provisional rationale: Longs Drugs passage says locations are in Hawaii; Warren Bryant passage identifies him as CEO.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> Which state does the drug stores, of which the CEO is Warren Bryant, are located?

**Reference answer**

> Hawaii

**Cited / retrieved passage text**

```text
[c2 | doc=Longs Drugs]
Longs Drugs is an American chain with approximately 40 drug stores throughout the state of Hawaii.

[c7 | doc=Warren Bryant]
Warren Bryant was the CEO of Longs Drugs Store Corporation out of California prior to the retail chain's acquisition by CVS/Caremark.  Hired in 2002 to Longs, he was Senior Vice President of The Kroger Company. , a retail grocery chain, from 1999 to 2002.  Prior to that, from 1996 to 1999, he was President and Chief Executive Officer of Dillon Companies, Inc., a retail grocery chain and subsidiary of The Kroger Co.  He is also a director of OfficeMax Incorporated.
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K02. source_id=5adf732a5542993a75d264e9

- Source dataset: `hotpotqa`
- Provisional S2 read: **faithful**
- Provisional rationale: Sue Donahue passage says Donahue replaced Kelli Ward.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> Which American politician did Donahue replaced

**Reference answer**

> Kelli Ward

**Cited / retrieved passage text**

```text
[c4 | doc=Sue Donahue]
Sue Donahue is an American politician.  Donahue was appointed in 2016 to serve in the Arizona State Senate representing the fifth legislative district as a member of the Republican Party.  Donahue replaced Kelli Ward who resigned to run for the United States Senate.  Donahue did not run for re-election in 2016 and was replaced by Sonny Borrelli.

[c7 | doc=Kelli Ward]
Kelli Ward ("née" Kaznoski; born January 25, 1969) is an American politician and osteopathic physician.  A member of the Republican Party, Ward was elected in 2012 to serve in the Arizona State Senate representing the fifth legislative district.  She was unopposed for election in 2014.  While serving in office, she continued to practice medicine in the emergency departments in Lake Havasu City and Kingman, Arizona.
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K03. source_id=5ac2a912554299218029dae8

- Source dataset: `hotpotqa`
- Provisional S2 read: **faithful**
- Provisional rationale: Wolfhounds formed in 1985; Hole formed in 1989 in Courtney Love passage.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> Which band was founded first, Hole, the rock band that Courtney Love was a frontwoman of, or The Wolfhounds?

**Reference answer**

> The Wolfhounds

**Cited / retrieved passage text**

```text
[c4 | doc=The Wolfhounds]
The Wolfhounds are an indie pop/noise pop band formed in Romford, UK in 1985 by Dave Callahan, Paul Clark, Andy Golding, Andy Bolton and Frank Stebbing, and originally active until 1990.  The band reformed in 2005 and continues to write, record and play live, releasing new albums in 2014 and 2016.

[c9 | doc=Courtney Love]
Courtney Michelle Love (born Courtney Michelle Harrison; July 9, 1964) is an American singer, songwriter, actress, and visual artist.  Prolific in the punk and grunge scenes of the 1990s, Love has enjoyed a career that spans four decades.  She rose to prominence as the frontwoman of the alternative rock band Hole, which she formed in 1989.  Love has drawn public attention for her uninhibited live performances and confrontational lyrics, as well as her highly publicized personal life following her marriage to Kurt Cobain.
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K04. source_id=5a7272eb5542997f827839d7

- Source dataset: `hotpotqa`
- Provisional S2 read: **faithful**
- Provisional rationale: Hunger Games passage identifies Katniss as 16-year-old and Catching Fire continues that story.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> How old is the female main protagonist of Catching Fire?

**Reference answer**

> 16-year-old

**Cited / retrieved passage text**

```text
[c0 | doc=Catching Fire]
Catching Fire is a 2009 science fiction young adult novel by the American novelist Suzanne Collins, the second book in "The Hunger Games trilogy".  As the sequel to the 2008 bestseller "The Hunger Games", it continues the story of Katniss Everdeen and the post-apocalyptic nation of Panem.  Following the events of the previous novel, a rebellion against the oppressive Capitol has begun, and Katniss and fellow tribute Peeta Mellark are forced to return to the arena in a special edition of the Hunger Games.

[c4 | doc=The Hunger Games (novel)]
The Hunger Games is a 2008 dystopian novel by the American writer Suzanne Collins.  It is written in the voice of 16-year-old Katniss Everdeen, who lives in the future, post-apocalyptic nation of Panem in North America.  The Capitol, a highly advanced metropolis, exercises political control over the rest of the nation.  The Hunger Games is an annual event in which one boy and one girl aged 12–18 from each of the twelve districts surrounding the Capitol are selected by lottery to compete in a televised battle to the death.
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K05. source_id=5ac2c3545542990b17b1548b

- Source dataset: `hotpotqa`
- Provisional S2 read: **faithful**
- Provisional rationale: Korea under Japanese rule ended at conclusion of World War II; Chang Ucchin born under that rule.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> Chang Ucchin was born in korea during a time that ended with the conclusion of what?

**Reference answer**

> World War II

**Cited / retrieved passage text**

```text
[c4 | doc=Korea under Japanese rule]
Korea under Japanese rule began with the end of the short-lived Korean Empire in 1910 and ended at the conclusion of World War II in 1945.  Japanese rule of Korea was the outcome of a process that began with the Japan–Korea Treaty of 1876, whereby a complex coalition of the Meiji government, military, and business officials sought to integrate Korea both politically and economically into the Empire of Japan.  A major stepping-stone towards the Japanese occupation of Korea was the Japan–Korea Treaty of 1905, in which the then-Korean Empire was declared a protectorate of Japan.  The annexation of Korea by Japan was set up in the Japan–Korea Treaty of 1910, which was never actually signed by the Korean Regent, Gojong.

[c9 | doc=Chang Ucchin]
Chang Ucchin (26 November 1917 – 27 December 1990) is one of the representatives of modern Korean fine art.  Chang was born when Korea was still under Japanese colonial rule.  He studied western art at Tokyo's Imperial School of Art.  He became a professor of fine arts at Seoul National University in 1954, but resigned to paint full-time from 1960.
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K06. source_id=ALCE-data/qampari_eval_gtr_top100:481__wikitables_simple__dev

- Source dataset: `alce`
- Provisional S2 read: **needs human check**
- Provisional rationale: List answer is long; sampled context supports science-fiction-magazine topic but full list needs manual verification.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> What magazine is a science fiction magazine ?

**Reference answer**

> Amazing Stories, Analog Science Fiction and Fact, Apex Magazine, [[Asimovs Science Fiction]], Clarkesworld Magazine, Galaktika, Galaxy Science Fiction, Interzone, Lightspeed, Locus, Mithila Review, Science Fantasy, SciFiNow, Sci Phi Journal, SFX, Tor.com, Kalpabiswa.

**Cited / retrieved passage text**

```text
[c0 | doc=5507536]
might be used, but until the appearance of a magazine specifically devoted to sf there was no need of a label to describe the category. The first specialized English-language pulps with a leaning towards the fantastic were "Thrill Book" (1919) and "Weird Tales" (1923), but the editorial policy of both was aimed much more towards weird-occult fiction than towards sf." Major American science fiction magazines include "Amazing Stories", "Astounding Science Fiction", "Galaxy Science Fiction", "The Magazine of Fantasy & Science Fiction" and "Isaac Asimov's Science Fiction Magazine". The most influential British science fiction magazine was "New Worlds"; newer British SF
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K07. source_id=ALCE-data/qampari_eval_gtr_top100:439__wikidata_intersection__dev

- Source dataset: `alce`
- Provisional S2 read: **needs human check**
- Provisional rationale: List answer is long; requires checking every film against script/producer constraints.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> For which movie did Mani Ratnam work on the script and serve as producer?

**Reference answer**

> Dil Se.., Thiruda Thiruda, Raavan, O Kadhal Kanmani, Kaatru Veliyidai, Iruvar, Kadal, Bombay, Sathriyan, Dumm Dumm Dumm.

**Cited / retrieved passage text**

```text
[c0 | doc=1732375]
Mani Ratnam Gopala Ratnam Subramaniam (born 2 June 1956), commonly known by his screen name Mani Ratnam, is an Indian film director, screenwriter, and producer who predominantly works in Tamil cinema. The Government of India honoured him with the Padma Shri, acknowledging his contributions to film in 2002. Despite being born into a film family, Mani Ratnam did not develop any interest towards films when he was young. Upon completion of his post graduation in management, he started his career as a consultant. He entered the film industry through the 1983 Kannada film "Pallavi Anu Pallavi". The failure of his
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K08. source_id=ALCE-data/qampari_eval_gtr_top100:772__wikidata_simple__dev

- Source dataset: `alce`
- Provisional S2 read: **needs human check**
- Provisional rationale: List answer names German Type U 31 examples; likely supported but all items need manual verification.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> What is an example of the marine vessel classification German Type U 31?

**Reference answer**

> SM U-31, SM U-33, SM U-37, SM U-41, SM U-38, SM U-34, SM U-35, SM U-40, SM U-39, SM U-36, SM U-32.

**Cited / retrieved passage text**

```text
[c0 | doc=10074607]
German Type U 31 submarine U 31 was a class of U-boats built during World War I by the Kaiserliche Marine. Between 1912 and 1915 eleven were built on Germaniawerft in Kiel, amongst these top-three-scoring with the famous Lothar von Arnauld de la Perière as commander, with Walter Forstmann and with Max Valentiner. Together these U-boats sunk more than . Later Admiral and head of the Abwehr from 1935 to 1944 Wilhelm Canaris also served as commander on two different U 31 U-boats. He first took over from Max Valentiner on "U-38" and later on . German Type U 31
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K09. source_id=ALCE-data/qampari_eval_gtr_top100:336__wikidata_comp__dev

- Source dataset: `alce`
- Provisional S2 read: **needs human check**
- Provisional rationale: List of countries is long and duplicate-heavy; manual verification needed.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> From which country did Seattle Storm make draft selections?

**Reference answer**

> Australia, United States of America, Romania, Turkey, United States of America, Hungary, United States of America, United States of America, Australia, Hungary, United States of America, Nigeria, United States of America, United States of America, United States of America, Israel, United States of America, United States of America, France.

**Cited / retrieved passage text**

```text
[c0 | doc=18461934]
attend the draft. Unless indicated otherwise, all players listed are Americans who played at U.S. colleges. 2015 WNBA draft The 2015 WNBA draft was the league's draft for the 2015 WNBA season. The Seattle Storm picked first. This year's draft was unusual in that two players with remaining college eligibility declared for the draft—Jewell Loyd of Notre Dame, who had one year of eligibility remaining, and Amanda Zahui B. of Minnesota, with two years remaining. Both players were draft-eligible by virtue of their births in 1993; under current draft rules, players who turn 22 in the calendar year of the
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

### K10. source_id=ALCE-data/qampari_eval_gtr_top100:479__wikitables_simple__dev

- Source dataset: `alce`
- Provisional S2 read: **needs human check**
- Provisional rationale: Bog-body list answer needs item-by-item support check.
- Human label: [ ] faithful  [ ] not faithful  [ ] unsure
- Human notes:

**Query**

> What are the discovered bog bodies ?

**Reference answer**

> Elling Woman, Grauballe Man, Haraldskær Woman, Huldremose Woman, Tollund Man, Kayhausen Boy, Osterby Man, Cashel Man, Gallagh Man, Old Croghan Man, Weerdinge Men, Yde Girl, Lindow Woman, Lindow Man, Cladh Hallan Skeletons.

**Cited / retrieved passage text**

```text
[c0 | doc=980456]
found in the Northern European countries of Denmark, Germany, the Netherlands, the United Kingdom and Ireland. In 1965, the German scientist Alfred Dieck cataloged more than 1,850 bog bodies, but later scholarship revealed much of the Dieck's work was erroneous, and an exact number of discovered bodies is unknown. Several bog bodies are notable for the high quality of their preservation and the substantial research by archaeologists and forensic scientists. These include: Bog body A bog body is a human cadaver that has been naturally mummified in a peat bog. Such bodies, sometimes known as bog people, are both geographically
```

**Human prompt**

> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.

