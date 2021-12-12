def arabize(src):
    tx = src.replace('<s>', '').replace('</s>', '').split(' ')
    rtl = tx[::-1]
    return ' '.join(rtl)

