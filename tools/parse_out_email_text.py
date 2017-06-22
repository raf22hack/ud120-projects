#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

try:
    import string
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(maketrans("", "", string.punctuation))

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)

        stemmer = SnowballStemmer("english")

        # print("this is the len: ", len(all_text))
        # print(content[1])
        #
        # for word in content[1].split(" "):
        #     print(word)

        stop_words = stopwords.words("english")


        bag_words = []

        for word in text_string.split():
            # if word not in stop_words:
            bag_words.append(stemmer.stem(word))

        # print(bag_words)

        string_bag = ' '.join(bag_words)

    return string_bag

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

