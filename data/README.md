# Book Dataset - Task 1

There are 46,226 books in 29 classes. This dataset contains book cover images, title, author, description and subcategories for each respective book.

Simplified training set and test set ground truth listed by image name and class number (listed below).

>book_descriptions_train.csv
> 
>book_descriptions_test.csv

Format:
```
[FILENAME] [DESCRIPTION] [CLASS NO.]
```

Example:
```
0805005021.jpg,"This is the only comprehensive volume of ...",14
1626400229.jpg,"Clifton’s Cafeteria―it might just be the ...",1
```

### /full_data


Training set and test set with all attributes including image URL, title, author, description and category.

>book_descriptions_train_full.csv

Contains all the entries with descriptions we have.

---

>book_descriptions_train_balanced.csv

Contains the same number of entries with descriptions for each category (balanced for training testing).

---

Format:
```
"[AMAZON INDEX (ASIN)]","[FILENAME]","[IMAGE URL]","[TITLE]","[AUTHOR]","[CATEGORY ID]","[CATEGORY]","[DESCRIPTION]","[STATUS_CODE]"
```

Example:
```
"0785829954","0785829954.jpg","http://ecx.images-amazon.com/images/I/51LK-aBfjSL.jpg","Weapons of War Bombers & Transport Aircraft 1939-1945",,"0","Arts & Photography","Part of the Weapons of War series, this book features more than ...","200"
"0823025330","0823025330.jpg","http://ecx.images-amazon.com/images/I/51mjx4REH2L.jpg","Illuminated Landscape","Peter Poskas","0","Arts & Photography","Shows examples of urban and rural landscapes, discusses the composition, light, and colors of ...",200
```

### /stats

Contains some statistic about the training and test set.

### /images

Contains all images.

### Category ID

29 classes

Training - 41,470 Total

Test - 4,756 Total

~ 90% train - test split

| Label |Category Name|Training Size|Test Size|
|-------|---|---|---|
| 0     |Arts & Photography|1430|164|
| 1     |Biographies & Memoirs|1430|164|
| 2     |Business & Money|1430|164|
| 3     |Children's Books|1430|164|
| 4     |Comics & Graphic Novels|1430|164|
| 5     |Computers & Technology|1430|164|
| 6     |Cookbooks, Food & Wine|1430|164|
| 7     |Crafts, Hobbies & Home|1430|164|
| 8     |Christian Books & Bibles|1430|164|
| 9     |Engineering & Transportation|1430|164|
| 10    |Health, Fitness & Dieting|1430|164|
| 11    |History|1430|164|
| 12    |Humor & Entertainment|1430|164|
| 13    |Law|1430|164|
| 14    |Literature & Fiction|1430|164|
| 15    |Medical Books|1430|164|
| 16    |Mystery, Thriller & Suspense|1430|164|
| 17    |Parenting & Relationships|1430|164|
| 18    |Politics & Social Sciences|1430|164|
| 19    |Reference|1430|164|
| 20    |Religion & Spirituality|1430|164|
| 21    |Romance|1430|164|
| 22    |Science & Math|1430|164|
| 23    |Science Fiction & Fantasy|1430|164|
| 24    |Self-Help|1430|164|
| 25    |Sports & Outdoors|1430|164|
| 26    |Teen & Young Adult|1430|164|
| 27    |Test Preparation|1430|164|
| 28    |Travel|1430|164|
