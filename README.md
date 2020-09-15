# Recipe Remix - Tensorflow Model
Have you ever wanted to make your own recipe? Well, now you can with this
Tensorflow model. This model has been trained to accept a string that could
be the start of your recipe, and complete it with its own suggestions!

This model was originally developed for the Bon-Hacketit Virtual Hackathon as
a part of the [Recipe Remix](https://devpost.com/software/recipe-remix?ref_content=user-portfolio&ref_feature=in_progress)
submission. Check out the rest of the team's work there!

## Technical Description
This model was trained for 30 epochs using the Palmetto Cluster and the
[Food.com Kaggle Dataset](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions). Based
on a tutorial on the Tensorflow website for developing text generation models, we chose to use a model that consisted
of an embedding layer to vectorize the text into a vector of 256 dimensions. From there, we used an LSTM layer
that would have the advantage of learning patterns on a character-by-character basis. While this design did allow for
the model to devlop its own words and foods, this did mean that it had to learn first what a word was, followed by
what a sentence was. This layer consisted of 1024 units that then outputted into a densely connected layer of neurons.

As I mentioned earlier, this model used a character-by-character approach for generating text. This means that the model
is able to develop its own vocabulary in some cases, such as when in the example output where it defines its own tool
called a *precubowl* which is meant to be a version of zucchini. Once again, the model was never provided with explicit
definitions of zucchini. Instead, it first had to learn that zucchini was a type of noun, before then determining that it
was a food that could be added. With further training, it would eb interesting to see what other connections or developments
the model could produce.

## Prerequisites
- Tensorflow 2
- Python 3

## Usage
To run the model, simply clone this repository to your machine and run
```bash
python test_script.py "input"
```
The model will print its output as well as save it to `output.txt`

### Example Output:
```
python test_script.py "add butter"
add butter and peanut butter in a frypan. add some cream , and baking sodas bring to a boil and simmer for 5 towels to drain. cook the pasta until tender , 4 level than. while spinach is quite sticky stand mixer , using an electric mixture and stir until gelatin is completely melted. stir in sour cream , and let sit button onions , peppers or raw chocolate chips and cumin. cook until golden brown and clean and add zucchini precubowl and gently stir in the bakingmix all of the ingredients into a medium saucepan , cook small piece of two favorite saucepreheat oven to 400. pierce potatoes in a pan , and add salt , & pepper in a shallow bowl. on high speed until well incorporated to taste. serve razed butter , mix well and continue baking for about 10 minutes , remove from heat. wash and dry the zucchini that measure whisk eggs. beat egg whites and curry. if you don't want the meat. reduce , bake barely tender. place sherry , chili powder , and cumin. cook over low heat until nd prepared. serve 1 /
```
```
python test_script.py "toast"
toast and cook , stirring occasionally , until just tender. add two tablespoons of olive oil and heat the saute zucchini to before serving add more heavy cream. pour this mixture into mixing bowl. chop the onion and garlic. cook until zucchini is cooked in the centre comes out with most more. best at the top. dredge chicken pieces on all sides of grease. season with salt and tep with parchment paper. mash coals. add the worcestershire saucepreheat the oven to 350 degrees. spread the remaining 2 tablespoons oil in the wok and mix in walnuts , then cheese. mix well before transferraie to chill cheeses and 3 cups. chop the prosting to distribute one partially 4-5 minutes more. pre-heat oven to 400f. in a broil 1 minute. add the potato until golden brown or nuts. transfer broccoli mixture. cook 2 to 3 minutes until the or square of foil. place sugar , salt and pepper. cook for 5 minutes , stirring constantly , and mix well. boil 5 minutes. if necessary , mix the eggs and pineapple and cut an in
```
```
python test_script.py "mix batter and eggs"
mix batter and eggs and mayonnaise and saute them alternately with a wonderfully and if desired , 2 minutes. cut tomatoes with a sprinkling the wash into a small batches until just tender. add flour. stir in butter. cook zucchini for 3-5 minutes per side , until browned and crispy and mushrooms are soft. add garlic and parm mixture. mix to combine. to make the cookies to a wire rack to cool. slice the jars in a stewer and pinch and sprinkle with the lemon zest to saucepan and cook 1 minute on high using a very well-combined. knead for one to the boiling coconut. heat a grill , mushrooms and toss with 3 / 4 cup of atld flour until smooth. add vanilla , baking powder and soda. add to creamed mixture. in another bowl , mix 1 / 4 cup vegetabixture. add flour and crumbs. sprinkle over some little you have chosen , developscombine all ingredients. bake at 350 f for 15 to 20 minutesstep 4processor lightly brown. meanwhile , prepare the potato side face down together well. pour americand floured 10-inch springfo
```
```
python test_script.py "add"
add all f if you would like it ngredients and blend wet. better the eggs. mix sliced zucchini in oil over medium heat for 5 minutes or until meat thermometer in the center. leave a curry paste and a pinch of grated potatoes:. in a 325-22-inch nonstick skillet over medium-high heat. add the mushrooms. bring to a simmer and cook 20 minutes. til beans are tender but. i like to let excess water covers almost to room temperature or if you like cooks. add milk , mix well and finish with the syrup except scallions. sprinkle with the remaining diced mushrooms and return them to pot. cover chicken breasts. drop by teaspoonsful , or run until no longer pink , 5-10 minutes , loosened and fragrant , 4 tablespoons of sauce or pot pumpkin , strosprinkle the flour and tarragon mustard. in separate greased shallow dish combine reason powder , ginger , salt and 1 / 4 cup beer. mix mayonnaise with cookie sheet and bake for 30 minutes until firm , add to a medium size pan , brown butter in a spray with non
```

