def make_cate_dic(df):
    df = df.rename(columns = lambda x:x.upper())
    df['depth1'] = df['GOODS_CATE'].apply(lambda x:x.split('>')[1].replace('(#M)',''))
    df['depth2'] = ''
    len_ = len(df)
    for i in range(len_):
        try:
            df['depth2'].values[i] = df['GOODS_CATE'].values[i].split('>')[2]
        except:
            df['depth2'].values[i] = df['GOODS_CATE'].values[i].split('>')[1].replace('(#M)','')

    df = df[['depth1','depth2']].drop_duplicates()
    MultiIndex = pd.DataFrame(df.groupby(['depth1','depth2'])['depth2'].count()).index
    tuple_ = tuple((x,y) for y,x in MultiIndex[:])
    cate_list = pd.DataFrame(tuple_, columns = ['depth2','depth1'])
    return cate_list
