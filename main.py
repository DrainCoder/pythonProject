import random
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def clean(text):
    clean_text = ''
    for ch in text.lower():
        if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя ':
            clean_text = clean_text + ch
    return clean_text

with open('/content/BOT_CONFIG.json') as f:
    BOT_CONFIG = json.load(f)
    del BOT_CONFIG['intents']['price']
    len(BOT_CONFIG['intents'].keys())

texts = []
y = []
for intent in BOT_CONFIG['intents'].keys():
    for example in BOT_CONFIG['intents'][intent]['examples']:
        texts.append(example)
        y.append(intent)
    len(texts), len(y)

train_texts, test_texts, y_train, y_test = train_test_split(texts, y, random_state=42, test_size=0.2)

# vectorizer = CountVectorizer(ngram_range=(1,3), analyzer='char_wb') #TfidfVectorizer(ngram_range=(1,5), analyzer='char_wb')
vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=0.1, max_df=0.7, 
                             max_features=100, analyzer='char_wb') 
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

vocab = vectorizer.get_feature_names_out()
len(vocab)

# clf = RandomForestClassifier(n_estimators=300) #LogisticRegression().fit(X_train, y_train)
# clf = AdaBoostClassifier(n_estimators=5000, random_state=0)
clf = KNeighborsClassifier(n_neighbors=1, weights = 'distance', 
    algorithm='brute', leaf_size=30, p=1,
    metric='minkowski', metric_params=None, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_train, y_train), clf.score(X_test, y_test) #LogisticRegression: 0.14893617021276595, RandomForestClassifier: 0.19574468085106383

def get_intent_by_model(text):
    return clf.predict(vectorizer.transform([text]))[0]

def bot(input_text):
    intent = get_intent_by_model(input_text)
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])

input_text = ''
while input_text != 'stop':
    input_text = input()
    if input_text != 'stop':
        response = bot(input_text)
        print(response)

import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    input_text = update.message.text
    output_text = bot(input_text)
    update.message.reply_text(output_text)


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("5010629338:AAGRKszcyoCUF067nljwOiR7YVGaDmBttJo")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()