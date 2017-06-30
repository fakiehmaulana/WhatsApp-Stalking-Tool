# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 19:46:10 2017

@author: LL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set(style="ticks")

color = sns.color_palette("coolwarm", 7)



# TODO 0 : transform the raw export .txt file into the csv format as below
# Desired format : datetime, member, content, lettercount, wordcount
filename = 'AnnivYi.txt'
filename = 'SalutFred.txt'
df0 = pd.read_table(filename, 
                    header = None, 
#                    skiprows = [0, 1],
                    names = ['test'])
# creating column 0 : datetime, column 1 : 
df0 = df0['test'].str.split(': ',  expand = True, n = 2)
df0.columns = ['datetime', 'member', 'content']


# Problem of multi-line message
dtt_format = r'[0-9][0-9]\/[0-9][0-9]\/[0-9][0-9][0-9][0-9]\ [0-9][0-9]\:[0-9][0-9]\:[0-9][0-9]'
#pd.Series(
#        ['31/05/2017 16:36:10', 
#         ' 31/05/2017 16:36:10', 
#         'sd 31/05/2017']
#        ).str.match(
#                dtt_format
#                )
# Select lines not starting with datetime format
df0['datetime'] = df0.astype('str')
df0[~df0['datetime'].str.match(dtt_format)] = df0[
        ~df0['datetime'].str.match(dtt_format)
        ].assign(content = ' ' + df0.datetime,
        datetime = None, member = None)
df0 = df0.fillna(method = 'pad')
df0 = df0.groupby(['datetime', 'member']).sum().reset_index() #concatenate same message
# hyp : only very few messages are sent at exactly the same second
# and among them, there are even less that are with skipped lines
# MAYBE : split with member before then fill / groupby with member to avoid errors


# création : 03/12/2016 17:56:59: ‎Vous avez créé le groupe “Anniv Marion&JL”
# 03/12/2016 17:57:00: ‎Les messages envoyés dans ce groupe sont désormais protégés avec le chiffrement de bout en bout.
# ajout de membre : ajouté
# membre qui quitte : quitté
# membre retiré : retiré
# icône changé : changé l'icône
# icône supprimée : supprimé l'icône
# sujet du groupe remplacé : sujet
# changement de numéro : est passé, est devenu
sys_msg = df0[
                df0['member'].str.contains("créé le groupe")
                | df0['member'].str.contains(
                        "chiffrement de bout en bout"
                        )
                | df0['member'].str.contains("ajouté")
                | df0['member'].str.contains("quitté")
                | df0['member'].str.contains("retiré")
                | df0['member'].str.contains("changé l'icône")
                | df0['member'].str.contains("supprimé l'icône")
                | df0['member'].str.contains("sujet")
                | df0['member'].str.contains("est passé")
                | df0['member'].str.contains("est devenu")
            ].reset_index()[['datetime', 'member']]
sys_msg.columns = ['datetime', 'content']
sys_msg['datetime'] = pd.to_datetime(sys_msg['datetime'], dayfirst = True)
sys_msg = sys_msg.sort_values('datetime', ascending = True)
sys_msg

df0 = df0[
            ~ (
                    df0['member'].str.contains("créé le groupe")
                    | df0['member'].str.contains("chiffrement")
                    | df0['member'].str.contains("ajouté")
                    | df0['member'].str.contains("quitté")
                    | df0['member'].str.contains("retiré")
                    | df0['member'].str.contains("changé l'icône")
                    | df0['member'].str.contains("supprimé l'icône")
                    | df0['member'].str.contains("sujet")
                    | df0['member'].str.contains("est passé")
                    | df0['member'].str.contains("est devenu")
                )
        ]

def msg_type(msg):
    if msg == '<‎image absente>':
        return('image')
#    if pd.Series(msg).str.contains('<vidéo absente>').iloc[0]:
    if msg == '<‎vidéo absente>':
        return('video')
    elif msg == '<‎audio omis>':
        return('audio')
    elif msg == '<GIF retiré>':
        return('gif')
    else : 
        return('text')
#df0['type'] = df0['content'].apply(msg_type)
df0['content'] = df0['content'].astype('str')
df0[df0.content == '<image absente>'] # work
df0[df0.content == '<vidéo absente>'] # does not work
df0.query("content == '<vidéo absente>'")
df0[df0.type != 'text']
# Switch content for non text message to None


#def fxy(x, y):
#    return x * y
#df['newcolumn'] = df.apply(lambda x: fxy(x['A'], x['B']), axis=1)


df0['lettercount'] = df0['content'].str.len()

# import preprocessed .csv chat history
filename = "AnnivYiexport.csv"
filename = "SalutFredexport.csv"
chatname = filename.replace('.csv', '')
df = pd.read_csv(filename, 
                 header = 0,
                 parse_dates = ['datetime'], 
                 dayfirst = True,
                 index_col = 'datetime')
                 # index = datetime, 
                 # cols : timestamp, member, type, wordcount, lettercount



# Plot total word count, letter count per person
print("Part 1 : Plot total word count, letter count per person")
grouped = df.groupby('member')
grouped[['member', 'wordcount', 'lettercount']].describe()
grouped[['member', 'wordcount', 'lettercount']].sum()

sizes = grouped[['member', 'wordcount']].sum().sort_values('wordcount')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, autopct='%1.1f%%', labels = sizes.index, startangle=90)
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Salut Fred wordcount')
plt.show()

sizes = grouped[['member', 'lettercount']].sum().sort_values('lettercount')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, autopct='%1.1f%%', labels = sizes.index, startangle=90)
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Salut Fred lettercount")
plt.show()

# can also be done using 
# grouped = df.groupby(['member', 'year', 'month', 'day'])
# grouped[['lettercount']].sum().groupby(level=0).sum()

# TODO : Group the least important ones
# github.com/gboeing/2014-summer-travels/blob/master/trip-visualization.ipynb 


# Plot total word count, letter count per person by datetime
print("Part 2 : total word count, letter count per person by datetime")
plt.figure()
for key, grp in df.groupby(['member']):
    print(key) # name of member
    #print(grp) # everything the member said
    print(grp['wordcount'])
    print(grp['wordcount'].tail(5))
    plt.plot(grp['wordcount'], label=key)
    #plt.plot(grp['wordcount'], index=grp['datetime'], label=key)
    #grp.plot(x='datetime', y='wordcount', label=key)
#    grp['D'] = pd.rolling_mean(grp['B'], window=5)    
#    plt.plot(grp['D'], label='rolling ({k})'.format(k=key))
plt.legend(loc='best')
plt.title('Wordcount per member')    
plt.show()

plt.figure()
for key, grp in df.groupby(['member']):
    plt.plot(grp['lettercount'], label=key)
#plt.legend(loc='best')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.title('Lettercount per member')    
plt.show()


# Plot total word count, letter count per day (or other scale)
# By using resample and sum : the scale can be changed
# Same result as doing group by using day month year but more efficient
print("Part 3 : total word count, letter count per day and per day per hour")

df2 = df.groupby(['member'])['lettercount'].resample('1D').sum().dropna(axis=0,
                how='any').reset_index()
# reset_index drops the multi-index
plt.figure()
for key, grp in df2.groupby(['member']):
    print(key)
    print(grp)
    plt.plot(grp['datetime'], grp['lettercount'], label = key,
             linestyle = 'dashed', marker = 'o')
#plt.legend(loc='best')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
           fancybox=False, shadow=False, ncol=4)
plt.title('Lettercount per day')
plt.show()

# One graph for each member
for key, grp in df2.groupby(['member']):
#    print(key) # name of member
#    print(grp) # df2[df2.member == key]
    plt.figure()
    plt.plot(grp['datetime'], grp['lettercount'], label = key,
             linestyle = 'dashed', marker = 'o')
    plt.legend(loc = 'best')
    plt.title("Lettercount of "+ key)
    plt.show()



# Total number of days, messages, words, letters, files
print("Part 4 : General information on different types of messages\n")
# Find earliest and latest message, compute difference
print(chatname, "conversation history",
      "\nfrom", df.index.min(), "to", df.index.max(), 
      "\nlasted", df.index.max()-df.index.min())

# count number of text, image and video messages

# Pie chart version
sizes = df.groupby('type')[['type']].count().sort_values('type')
sizes.columns = ['Count']
plt.figure()
plt.pie(sizes, autopct='%1.1f%%', labels = sizes.index, startangle=90)
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Types of messages')
plt.show()

# Bar plot version
# Without labels
#plt.figure()
#plt.bar(range(len(sizes.index)), height = sizes['Count'], width = 0.35)
#plt.xticks(range(len(sizes.index)), sizes.index)
#plt.title('Types of messages')
#plt.ylabel('Number of messages')
#plt.show()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    
    Source : https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height, # '%d' % int(height), if int precision only
                ha='center', va='bottom')        
plt.figure()
barplot = plt.bar(range(len(sizes.index)), height = sizes['Count'], width = 0.35)
plt.xticks(range(len(sizes.index)), sizes.index)
plt.title('Messages sent on ' + chatname)
plt.ylabel('Number of messages')
autolabel(barplot)
plt.show()


# plot by member
df3 = df.groupby(['member','type'])[['type']].count() # by member
df3.columns = ['Count']
df3 = df3.reset_index()
n_member = df3['member'].drop_duplicates().count() # number of members
n_type = df3['type'].drop_duplicates().count() # number of types of messages
f, axarr = plt.subplots(n_type, 1) # Initialize figure
iterator = 0
for type_key, type_grp in df3.groupby(['type']):
    print(type_key)
    print(type_grp)
    axarr[iterator].pie(
            type_grp['Count'], 
            autopct='%1.1f%%', 
            labels = type_grp['member'])
    axarr[iterator].axis('equal')
    axarr[iterator].set_title("Proportion of " + type_key + " messages",
         fontsize = 12)
    iterator = iterator + 1
plt.show()
# Change into stacked bar plot
right = df3.groupby(['type'])[['Count']].sum()
right.columns = ['Type_count']
right = right.reset_index()
merged = pd.merge(df3, right, how = 'left', on = ['type'])
merged = merged.assign(proportion = merged.Count/merged.Type_count
                       )[['member', 'type', 'proportion']]
merged = merged.pivot(index = 'type',
                      columns = 'member', 
                      values = 'proportion')
merged.plot(kind = 'bar', 
            stacked = True, 
            title = 'Proportion of messages types')
#plt.legend(title = 'Member', loc = 'best')
plt.legend(title = "Member", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Message type")
plt.ylabel("Proportion")


# Group by member
# Pie chart version 
for member_key, member_grp in df3.groupby(['member']):
    plt.figure()
    plt.pie(member_grp['Count'], autopct='%1.1f%%', 
            labels = member_grp['type'])
    plt.axis('equal')
    plt.title("Messages from " + member_key)
    plt.show()
# Stacked bar plot version
right = df3.groupby(['member'])[['Count']].sum()
right.columns = ['Type_count']
right = right.reset_index()
merged = pd.merge(df3, right, how = 'left', on = ['member'])
merged = merged.assign(proportion = merged.Count/merged.Type_count
                       )[['member', 'type', 'proportion']]
merged = merged.pivot(index = 'member',
                      columns = 'type', 
                      values = 'proportion')
merged.plot(kind = 'bar', 
            stacked = True, 
            title = 'Proportion of messages types')
plt.legend(title = "Type", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Member")
plt.ylabel("Proportion")

# for only one member
selected_member = "Laurent Lin"
plt.figure()
plt.pie(df3[df3.member == selected_member]['Count'], 
        autopct='%1.1f%%', 
        labels = df3[df3.member == selected_member]['type'])
plt.axis('equal')
plt.title("Messages from " + selected_member)
plt.show()




# Most active days
df2.groupby(['datetime'])['datetime','lettercount'].sum().idxmax() 
# by member
df2.groupby(['member'])[['lettercount']].idxmax()
df2.ix[list(df2.groupby(['member'])[['lettercount']].idxmax()['lettercount'])]
# single member
df2.ix[df2[df2.member == selected_member]['lettercount'].idxmax()]['datetime']
# days were total number of messages > threshold


# Display messages from one period
print("Display messages from one period")
print("start_date and end_date format should be :")
print("YYYY-MM-DD")
print("end_date is included")
start_date = "2017-01-21"
end_date = "2017-01-22"
df[(df.index > pd.Timestamp(start_date)) & 
   (df.index < pd.Timestamp(end_date) + 
    pd.Timedelta('1 Days'))][['member', 'type', 'content']]


print("Part 5 : concerning average words")
# Average words, messages in total
word_bymember = grouped[['member', 
                           'wordcount']].sum(
                           ).sort_values('wordcount')
n_msg = df.groupby('type')[[
        'type']].count().sort_values('type').loc['text']
print("In total", n_msg, "messages")
print("Average number of words per message =", 
      float(word_bymember.sum()/int(n_msg)))
# Average word per message by member
msg_bymember = df.groupby(['type',
                           'member'
                           ])[['type']].count().sort_values('type')
msg_bymember.columns = ['msg_count']
msg_bymember = msg_bymember.reset_index()
msg_bymember = pd.merge(
        word_bymember.reset_index(), 
        msg_bymember[msg_bymember.type == 'text'].drop('type', axis = 1),
        how = 'left', on = ['member'])
msg_bymember = msg_bymember.assign(
        avg_word = msg_bymember.wordcount/msg_bymember.msg_count)

# Without autolabel
#plt.figure()
#msg_bymember_plot = msg_bymember.plot(x = 'member', y= 'avg_word', 
#                                      kind = 'bar', legend = False)
#plt.title("Average word per message")
#plt.xlabel("")
##plt.xticks(range(len(msg_bymember.index)), msg_bymember.member)
#plt.ylabel("Average word per message by members")
#plt.show()

# With autolabel
plt.figure()
barplot = plt.bar(range(len(msg_bymember.index)), 
                  height = msg_bymember['avg_word'], width = 0.35)
plt.title("Average word per message")
plt.xlabel("")
plt.xticks(range(len(msg_bymember.index)), 
           msg_bymember.member,
           rotation = 30)
plt.ylabel("Average word per message")
autolabel(barplot)
plt.show()


# TODO : avg message per day, per hour and the same per person, 
# same thing with wordcount


# Wordcount per day per member
df.groupby(['member'])[['wordcount']].resample('1D').sum().reset_index()
# Number of message per member per day
df5 = df.groupby(['member'])[['wordcount']].resample('1D').count()
df5.columns = ["Count"]
df5 = df5.reset_index()
# Average word per message per day
df4 = df.groupby(['member'])[['wordcount']].resample('1D').mean().reset_index()
# Plot as function of time
for key, grp in df4.groupby(['member']):
    plt.figure()
    plt.plot(grp['datetime'], grp['wordcount'], label = key,
             linestyle = ':', marker = '.')
    plt.title("Average word per message from " + key)
    plt.show()
# Boxplot
plt.figure()
sns.boxplot(x = "member", y = "wordcount", data = df4, palette = sns.color_palette("Set2")).set_title(
        "Boxplot of average word per message per day")
plt.xticks(rotation = 30)
sns.despine(offset=10, trim=True)
plt.show()

# Boxplot with messages per day
plt.figure()
sns.boxplot(x = "member", y = "Count", data = df5)
plt.title("Boxplot of number of messages per day")
plt.xticks(rotation = 30)
sns.despine(offset=10, trim=True)
plt.show()

# Stats of wordcount
# Histogram
for key, grp in df.groupby(['member']):
    plt.figure()
    grp[["wordcount"]].hist(bins =100)
    plt.title(key + "\'s wordcount repartition")
    plt.show()
# Boxplot
sns.boxplot(x = 'member', y = 'wordcount', 
            data = df).set_title("Wordcount per message")

print("Part 6 : concerning time laps between messages")
# Sort to avoid negative bides
df_sorted = df.reset_index().sort_values(['datetime'], ascending = True)

shifted = df_sorted.shift(-1)[["datetime", 'content', 'member', 'type']]
shifted.columns = ["shifted_datetime", "msg_postbide",
                   "member_postbide", 'type_postbide']
shifted = pd.concat([df_sorted, shifted], axis = 1)
shifted = shifted.assign(bide = shifted.shifted_datetime - shifted.datetime)
shifted = shifted.assign(auto_debidage = 
                         (shifted.member == shifted.member_postbide))
# auto_debidage : True if following message is from same user
shifted = shifted.assign(s_bide = shifted.bide.astype('timedelta64[s]'))
# search largest and smallest bides
shifted.nlargest(50, 'bide')[
        ['datetime', 'member', 'bide', 'content', 
         'member_postbide', 'msg_postbide', 'type_postbide', 'auto_debidage']]    
bide_min = pd.Timedelta('5 hours')
shifted[shifted.bide > bide_min].sort_values(['bide'], ascending = False)
shifted.nsmallest(10, 'bide')[
        ['datetime', 'shifted_datetime',
         'member', 'bide', 'content', 
         'member_postbide', 'msg_postbide', 'type_postbide', 'auto_debidage']]    
# General information
shifted['bide'].describe()
# Plot bides > bide_min
plt.figure()
sns.boxplot(x = 'member', y = 's_bide', 
            data = shifted[shifted.bide > bide_min])
sns.swarmplot(x = 'member', y = 's_bide', 
              data = shifted[shifted.bide > bide_min],
              color = '.25')
plt.title('Boxplot of bides > ' + str(bide_min))
plt.ylabel('bide in seconds')
plt.show()

# Number of auto_rep
# Consecutive messages = a_rep
shifted2 = df_sorted.shift(1)[["datetime", 'content', 'member', 'type']]
shifted2.columns = [
        "shifted_datetime", 
        "msg_prebide",
        "member_prebide", 
        'type_prebide'
        ]
shifted2 = pd.concat([df_sorted, shifted2], axis = 1)
shifted2 = shifted2.assign(
        auto_responding = (shifted2.member == shifted2.member_prebide)
        )
shifted2['block'] = (
        shifted2['auto_responding'] != shifted2['auto_responding'].shift(1)
        ).astype(int).cumsum()
shifted2['block_cumcount'] = shifted2[
        ['auto_responding', 'block']
        ].groupby('block').transform(lambda x: list(range(1, len(x) + 1)))
# https://stackoverflow.com/questions/25119524/pandas-conditional-rolling-count
shifted2 = shifted2.assign(
        a_rep = shifted2.auto_responding * shifted2.block_cumcount
        )   
shifted2[
        ['member', 'auto_responding', 'block', 'block_cumcount', 'a_rep']
        ].nlargest(50, 'a_rep')

shifted2['a_rep_count'] = shifted2[
        ['block', 
         'auto_responding']
        ].groupby(
        'block'
        ).transform(
                lambda x: len(x)
                )
shifted2 = shifted2.query(
        '(auto_responding == True)'
        ).drop_duplicates(
                subset = [
                'member', 'auto_responding', 
                'block', 'a_rep_count'])

shifted3 = shifted2.set_index(
        keys = 'datetime'
        ).groupby(
                ['member']
                )['a_rep_count'].resample('1D').sum().dropna(
                        axis=0, 
                        how='any'
                        ).reset_index()
plt.figure()
for key, grp in shifted3.groupby(['member']):
    print(key)
    print(grp)
    plt.plot(grp['datetime'], grp['a_rep_count'], label = key,
             linestyle = 'dashed', marker = 'o')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
           fancybox=False, shadow=False, ncol=4)
plt.title('Number of auto-responses per day')
plt.xlabel('Time')
plt.ylabel('Count')
plt.show()
# TODO : boxplot


print("Part 7 : Searching for keywords, emojis")
# TODO : most used emojis, word
import re
df['content'] = df['content'].str.lower()
pd.Series(
        ['1', 'lol', 'test', 'looool', 
           'je me suis', 'tu es là?',
           'mdrtr', 'lol loool', 'lol lol lol']
        ).str.contains('oo')
# Counts the times encountering pattern inside findall
pd.Series(
        ['1', 'lol', 'test', 'looool', 
           'je me suis', 'tu es là?',
           'mdrtr', 'lol loool', 'lol lol lol']
        ).str.findall('lol').str.len()
re.findall(r'\w*l*lo+\w*l+\w*',
           "looool loul lol haha, lllooll, troplolol, lololol, relol, loooolllll, lol, lolz, mdr, Lol， Loool")
pd.Series(
        ['1', 'lol', 'test', 'looool', 
           'je me suis', 'logiciel',
           'mdrtr', 'lol loool', 'lol lol lol',
           'looool loul lol', 'lllloooool', 
           'trolooolol', 'lololol', 'lllooolll',
           'Lol']
        ).str.findall(r'\w*l*lo+o*l+\w*')

df['lol'] = df['content'].str.findall(r'\w*l*lo+o*l+\w*').str.len()
df['lol_form'] = df['content'].str.findall(r'\w*l*lo+o*l+\w*')
df[df.lol > 0][['content', 'member', 'lol', 'lol_form']]
df['lol'].describe()

df_lol = df.groupby(['member'])['lol'].resample('1D').sum().dropna(axis=0, 
          how='any').reset_index()
df_lol.describe()
plt.figure()
for key, grp in df_lol.groupby(['member']):
    plt.plot(grp['datetime'], grp['lol'], label = key)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
           fancybox=False, shadow=False, ncol=4)
plt.title('Lol per day per member')
plt.show()

df_lol2 = df['lol'].resample('1D').sum().dropna(axis=0, how='any').reset_index()
df_lol2.describe()
plt.figure()
plt.plot(df_lol2['datetime'], df_lol2['lol'])
plt.xlabel('Time')
plt.ylabel('Count per day')
plt.title('"Lol" per day in ' + chatname)
plt.show()

sizes = df.groupby('member')[['lol']].sum()
sizes.columns = ['Lol percentage']
sizes = sizes.sort_values('Lol percentage')
plt.figure()
plt.pie(sizes, autopct='%1.1f%%', labels = sizes.index, startangle=90)
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Lol distribution')
plt.show()

plt.figure()
barplot = plt.bar(range(len(sizes.index)), height = sizes['Lol percentage'], width = 0.35)
plt.xticks(range(len(sizes.index)), sizes.index)
plt.title('Messages sent on ' + chatname)
plt.xlabel('Number of "Lol" in each message')
plt.ylabel('"Lol" distribution')
autolabel(barplot)
plt.show()
# TODO : most frequent form of "Lol"
df['lol_form'] = df['content'].str.findall(r'\w*l*lo+o*l+\w*')
df[df.lol > 0][['content', 'member', 'lol', 'lol_form']]


# TODO : emojis count
from collections import Counter
df6 = df
#df6['content'] = df6['content'].str.split(' ')
#df6.assign(wordocc = Counter(df6.content))
#dict(Counter(['apple','red','apple','red','red','pear']))
#Counter({'red': 3, 'apple': 2, 'pear': 1})
df6 = pd.DataFrame(df6['content'], columns = ['content'])
#df6.content.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)

content_sum = []
for msg in list(df6.dropna(axis=0, how='any')['content'].str.split(' ')):
    for word in msg:
        content_sum += word
content_sum = Counter(content_sum) # Count of every letter/emoji/chinese caracter 
# select only emojis
# but no word ??

#dict_you_want = { your_key: old_dict[your_key] for your_key in your_keys }
import emojis
emojis_list = emojis.generate_emojis()
for emoji in emojis_list:
    if not emoji in content_sum :
        emojis_list.remove(emoji)
        # remove an emoji that was not encountered 
        # but there are still emojis with 0 appearance...
len(emojis_list)
content_sum = {emoji : content_sum[emoji] for emoji in emojis_list}
emojis_list = pd.Series(content_sum, 
                        name = 'Count').sort_values(ascending = False)
emojis_list = pd.DataFrame(emojis_list).query('Count > 0')

print('10 most used emojis', emojis_list.head(10))
# Can't show emojis as labels...
plt.figure()
barplot = plt.bar(
        range(len(sizes.index)), 
        height = sizes['Count'], 
        width = 0.35)
plt.xticks(range(len(sizes.index)), sizes.index)
plt.title('Use of emojis in ' + chatname)
plt.ylabel('Number of appearances')
#autolabel(barplot)
plt.show()

# TODO : find better colormap

# TODO : compare user in two conversations