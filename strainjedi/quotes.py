from random import choice


def quotes():
    """
    returns random quote from star wars
    """

    list_of_quotes = [
        '"May the Force be with you." (Han Solo)',
        '"I am your father." (Darth Vader)',
        '"Use the Force, Luke." (Obi-Wan Kenobi)',
        '"Beep" (R2D2)',
        '"I felt a great disturbance in the force, as if millions of voices suddenly cried out in terror, '
        'and were suddenly silenced" (Obi-Wan Kenobi)',
        '"Never tell me the odds!" (Han Solo)',
        '"Well, you said you wanted to be around when I made a mistake." "...I take it back!" (Han Solo and Princess '
        'Leia)',
        '"You have your moments. Not many of them, but you do have them." (Princess Leia)',
        '"A long time ago in a galaxy far, far away..." (Opening)',
        '"Help me Obi-Wan Kenobi. You\'re my only hope." (Princess Leia)',
        '"Boy, it\'s lucky you have these compartments." "I use them for smuggling. I never thought I\'d be smuggling '
        'myself in \'em." (Luke Skywalker and Han Solo)',
        '"I love you." "I know." (Princess Leia and Han Solo)',
        '"You\'ve never heard of the Millennium Falcon? It\'s the ship that made the Kessel run in less than 12 '
        'parsecs." (Han Solo)',
        '"The Force is strong with this one." (Darth Vader)',
        '"I won\'t fail you. I\'m not afraid." "You will be. You... will... be..." (Luke Skywalker and Yoda)',
        '"Remember... The Force will be with you, always." (Obi-Wan Kenobi)',
        '"Judge me by my size, do you?" (Yoda)',
        '"RRRRRRRRRRR!" (Chewbacca)',
        '"Ben! I can be a Jedi. Ben, tell him I\'m ready!" (Thumps head on ceiling.) "Ow!" (Luke Skywalker)',
        '"Now, witness the power of this fully armed and operational battle station!" (Emperor Palpatine)',
        '"...Scoundrel. I like the sound of that." (Han Solo)',
        '"I have a bad feeling about this." (basically everyone)',
        '"Hmm! Adventure. Hmmpf! Excitement. A Jedi craves not these things." (Yoda)',
        '"Who\'s the more foolish; the fool, or the fool who follows him?" (Obi-Wan Kenobi)',
        '"You don\'t belive. That... is why you fail." (Yoda)',
        '"It could be worse." (Garbage creature growls) "It\'s worse." (Princess Leia and Han Solo)',
        '"That\'s no moon." (Obi-Wan Kenobi)',
        '"Sorry about the mess." (Han Solo)',
        '"Ready are you? What know you of ready? For eight hundred years have I trained Jedi. '
        'My own counsel will I keep on who is to be trained. A Jedi must have the deepest commitment, '
        'the most serious mind. This one, a long time have I watched. All his life has he looked away... to the '
        'future, to the horizon. Never his mind on where he was. ...Hmm? On what he was doing." (Yoda)',
        '"If they follow standard Imperial procedure, they\'ll dump their garbage before they go to light-speed. '
        'Then we just... float away." "...With the rest of the garbage." (Han Solo and Princess Leia)',
        '"Laugh it up, fuzzball!" (Han Solo)',
        '"I never doubted you! Wonderful!" (C-3PO)', '"You will never find a more wretched hive of scum and '
                                                     'villainy. We must be cautious." (Obi-Wan Kenobi)',
        '"Would somebody get this big walking carpet out of my way?!" (Princess Leia referring to Chewbacca)',
        '"No reward is worth this." (Han Solo)',
        '"I happen to like... nice men." (Princess Leia)',
        '"We would be honored if you would join us." (Darth Vader)',
        '"So what I told you was true... from a certain point of view." "...A certain point of view?!" (Obi-Wan '
        'Kenobi and Luke Skywalker)',
        '"Your weapons, you will not need them." (Yoda)',
        '"...Boring conversation anyway. Luke! We\'re gonna have company!" (Han Solo)',
        '"...I think I just blasted it." (Luke Skywalker)',
        '"Noooo! That\'s not true! That\'s impossible." (Luke Skywalker)',
        '"Search your feelings." (Emperor Palpatine)',
        '"I\'ll never join you." (Luke Skywalker)',
        '"He certainly has courage." "...Yeah, but what good is that if he gets himself killed?" (Princess Leia and '
        'Luke Skywalker)',
        '"You\'ve failed, your highness. I am a Jedi, as my father was before me." "...So be it, Jedi." (Luke '
        'Skywalker and Emperor Palpatine)',
        '"Only at the end do you realize the power of the Dark Side." (Emperor Palpatine)',
        '"It\'s not impossible. I used to bullseye womp rats in my T-16 back home, they\'re not much bigger than two '
        'meters." (Luke Skywalker)',
        '"Awww! But I was going into Tosche Station to pick up some power converters!" (Luke Skywalker)',
        '"It\'s a trap!" (Admiral Ackbar)',
        '"But how could they be jamming us if they don\'t know... that we\'re... coming?" (Lando Calrissian)',
        '"He is as clumsy as he is stupid." (Darth Vader)',
        '"If you strike me down, I will become more powerful than you could possibly imagine." (Obi-Wan Kenobi)',
        '"Stay on target!" (Gold Five)',
        '"...It\'s not fair! They promised me they fixed it! It\'s not my fault!" (Lando Calrissian)',
        '"You know, that little droid is going to cause me a lot of trouble." "...Oh, he excels at that, Sir." (Luke '
        'Skywalker and C-3PO)',
        '"If you\'re saying that coming here was a bad idea, I\'m starting to agree with you." (Luke Skywalker)',
        '"For over a thousand generations, the Jedi were the guardians of peace and justice in the Old Republic - '
        'before the dark times. Before the Empire." (Obi-Wan Kenobi)',
        '"Shut him up or shut him down." (Han Solo about C-3PO)',
        '"Give yourself to the Dark Side. It is the only way you can save your friends. Yes; your thoughts betray '
        'you. Your feelings for them are strong. Especially for your... sister. So, you have a twin sister. Your '
        'feelings have now betrayed her too. Obi-Wan was wise to hide her from me. Now, his failure is complete. If '
        'you will not turn to the Dark Side... then perhaps she will..." (Darth Vader)',
        '"I find your lack of faith disturbing." (Darth Vader)',
        '"Uh, we had a slight weapons malfunction, but uh... everything\'s perfectly all right now. We\'re fine. '
        'We\'re all fine here now, thank you." (Han Solo)',
        '"You are a member of the rebel alliance, and a traitor." (Darth Vader)',
        '"The circle is now complete." (Darth Vader)',
        '"Hey, I think my eyes are getting better. Instead of a big dark blur, I see a big bright blur." "...There\'s '
        'nothing to see. I used to live here, you know." "...You\'re gonna die here, you know. Convenient." (Han Solo '
        'and Luke Skywalker)',
        '"Why, you stuck up, half-witted, scruffy-looking Nerf herder." (Princess Leia)',
        '"Ungh. And I thought they smelled bad on the outside." (Luke Skywalker)',
        '"Would it help if I got out and pushed?!" "...It might!" (Princess Leia and Han Solo)',
        '"You don\'t have to do this to impress me." (Princess Leia)',
        '"Try not. Do... or do not. There is no try." (Yoda)',
        '"Luminous beings are we, not this crude matter." (Yoda)',
        '"All too easy." (Darth Vader)',
        '"How you doin\', Chewbecca? Still hanging around with this loser?" (Lando Calrissian)',
        '"I assure you, Lord Vader. My men are working as fast as they can." "Perhaps I can find new ways to motivate '
        'them." (Tiaan Jerjerrod and Darth Vader)',
        '"You\'ll find I\'m full of surprises!" (Luke Skywalker)',
        '"Yeah... you\'re a real hero." (Han Solo)',
        '"A Jedi Knight? Jeez, I\'m out of it for a little while, everyone gets delusions of grandeur!" (Han Solo)',
        '"I\'m Luke Skywalker! I\'m here to rescue you!" "...You\'re who?" (Luke Skywalker and Princess Leia)',
        '"Everything is proceeding as I have foreseen." (Emperor Palpatine)',
        '"Bounty hunters! We don\'t need this scum." (Admiral Piett)',
        '"Keep your distance, Chewie, but don\'t, y\'know, look like you\'re keeping your distance." (Grumbled '
        'questioning bark) "...I don\'t know. Fly casual." (Han Solo and Chewbacca)',
        '"What have you done?! I\'m BACKWARDS." (C-3PO)',
        '"You will find that it is you who are mistaken... about a great many things." (Emperor Palpatine)',
        '"When I left you, I was but the learner. Now I am the master." "Only a master of evil, Darth." (Darth Vader '
        'and Obi-Wan Kenobi)',
        '"We seem to be made to suffer. It\'s our lot in life." (C-3PO)',
        '"We have - ungh! - powerful friends. You\'re going to regret this." "...I\'m sure." (Princess Leia and Jabba '
        'the Hutt)',
        '"It\'s against my programming to impersonate a deity." (C-3PO)',
        '"These aren\'t the droids you\'re looking for." (Obi-Wan Kenobi)',
        '"Aren\'t you a little short for a Stormtrooper?" (Princess Leia)',
        '"Wait. I know that laugh..." (Han Solo)', '"This is some rescue!" (Princess Leia)',
        '"He\'s the brains, sweetheart!" (Han Solo, indicating Luke)',
        '"You are unwise to lower your defenses!" (Darth Vader)',
        '"Travelling through hyperspace ain\'t like dustin\' crops, boy!" (Han Solo)',
        '"They\'re getting closer." "...Oh yeah? Watch this." (Long pause ... the engine sputters and dies.) ('
        'Princess Leia and Han Solo)',
        '"Great, kid. Don\'t get cocky." (Han Solo)',
        '"You would prefer another target? A military target? Then name the system!" (Governor Tarkin)',
        '"R2-D2, you know better than to trust a strange computer!" (C-3PO)',
        '"Luke, you switched off your targeting computer. What\'s wrong?" "...Nothing! I\'m all right." (Yavin Base '
        'controller and Luke Skywalker)',
        '"So, what do you think? You think a princess and a guy like me-" (Han Solo)',
        '"I want them alive - no disintegrations." (Darth Vader)',
        '"I\'ve just made a deal that will keep the Empire out of here forever." (Lando Calrissian)',
        '"I saw... a city in the clouds." (Luke Skywalker)',
        '"Told you I did. Reckless is he. ...Now, matters are worse." (Yoda)',
        '"That boy is our last hope." (Obi-Wan Kenobi)', '"There\'s always a bigger fish." (Qui-Gon)'
    ]
    return choice(list_of_quotes)
