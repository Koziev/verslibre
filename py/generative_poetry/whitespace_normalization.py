import re


def normalize_whitespaces(s):
    s = s.replace('\u200c', ' ')

    # Уберем пробелы перед некоторыми знаками пунктуации
    for c in ',.?!…;:':
        s = s.replace(' ' + c, c)

    # Уберем ненужные пробелы у лапок
    s = s.replace('« ', '«').replace(' »', '»')

    # Уберем пробелы вокруг дефиса в MWE типа "кто-то"
    uline = s.lower()
    for w in 'когда кто кого чего где как куда откуда почему отчего зачем потому что чем кем кому ком чём чему чей чья чье чьё чьи чьих чьим чьей чью какой какая какое какие какого какую какие каких каким'.split():
        wx = [w]
        for i, c in enumerate(w):
            if c in 'уеыаоэёяию':
                w2 = w[:i] + c + '\u0301' + w[i+1:]
                wx.append(w2)

        for w2 in wx:
            if w2 in uline:
                s = re.sub('(' + w2 + ') - то', '\\1-то', s, flags=re.I)

    s = re.sub(r' - ка\b', '-ка', s)  # Глянь - ка ==> Глянь-ка
    s = re.sub(r'\bиз - за\b', 'из-за', s, flags=re.I)  # из - за ==> из-за
    s = re.sub(r'\bиз - под\b', 'из-под', s, flags=re.I)
    s = re.sub(r' - нибудь\b', '-нибудь', s, flags=re.I)  # Кто - нибудь ==> Кто-нибудь
    s = re.sub(r'\bпо - (.+?)\b', r'по-\1', s, flags=re.I)  # по - новому ==> по-новому
    s = re.sub(r'\bко́е - как\b', 'ко́е-как', s, flags=re.I)
    s = re.sub(r'\bкое - как\b', 'кое-как', s, flags=re.I)
    s = re.sub(r'\bко́е - что\b', 'ко́е-что', s, flags=re.I)
    s = re.sub(r'\bо́бщем - то\b', 'о́бщем-то', s)

    s = re.sub(r'(чу́?ть) - (чу́?ть)', r'\1-\2', s, flags=re.I)  # Чтоб задержаться на чуть - чуть...

    s = re.sub(r'(о́?чень) - (о́?чень)', r'\1-\2', s, flags=re.I)  # Очень - очень славный дед
    s = re.sub(r'(давны́?м) - (давно́?)', r'\1-\2', s, flags=re.I)  # Давным - давно, ты знаешь, все так было,

    return s


if __name__ == '__main__':
    print(normalize_whitespaces('Давным - давно, ты знаешь, все так было,'))
    print(normalize_whitespaces('Очень - очень славный дед,'))
    print(normalize_whitespaces('Чтоб задержаться на чуть - чуть...'))

    print(normalize_whitespaces('Давай - ка миленький мой, слазь'))
    print(normalize_whitespaces('по - новому работай'))
    print(normalize_whitespaces('из - под полы'))
    print(normalize_whitespaces('уснул из - за мороза'))

    print(normalize_whitespaces('Я когда - то пел'))
    print(normalize_whitespaces('Какой - то серый кот'))
    print(normalize_whitespaces('Но кто - нибудь придет'))
    print(normalize_whitespaces('Из - за кручи'))
    print(normalize_whitespaces('все бы́стро ка́к - то, впопыха́х'))
    print(normalize_whitespaces('а пла́тье гля́нь - ка уцеле́ло'))
