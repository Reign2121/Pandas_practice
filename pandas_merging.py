 # -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

"""# **데이터 합치기**

pandas 객체에 저장된 데이터는 여러 가지 방법으로 합칠 수 있다.

1. pandas.merge는 하나 이상의 키를 기준으로 DataFrame의 "로우"를 합친다. 관계형 데이터베이스의 join과 유사하다. merge는 키

2. pandas.concat 은 하나의 축을 따라 객체를 이어붙인다. concat은 축을 따라서

**A. Merge**

merge, join 연산은 관계형 데이터베이스의 핵심적인 연산으로서, 하나 이상의 키를 사용해서 데이터 집합의 로우를 합친다. pandas의 merge 함수는 이러한 알고리즘을 사용하고 있음.
"""

df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'],
                    'data':range(7)})
df2 = pd.DataFrame({'key':['a','b','d'],
                    'data2':range(3)})
print(df1, df2)

pd.merge(df1, df2) #중복되는 key를 기준으로 병합한다.

"""# **on='key'**

어떤 컬럼을 병합할 것인지 명시하지 않았으나, merge 함수는 중복된 컬럼 이름을 키로 사용한다. 

**하지만 명시적으로 지정해주는 습관을 들일 필요성이 있음**
"""

pd.merge(df1, df2, on='key')

"""**left_on, right_on**

만약 중복된 컬럼 이름이 없다면 left_on과 right_on으로 각각의 컬럼명을 적어 merge 할 수 있음
"""

df3 = pd.DataFrame({'lkey':['b','b','a','c','a','a','b'],
                    'data':range(7)})
df4 = pd.DataFrame({'rkey':['a','b','d'],
                    'data2':range(3)})


pd.merge(df3, df4, left_on='lkey', right_on='rkey') #이름을 직접 넣어주기도 하는군요

"""# **inner join**

merge 함수는 기본적으로 inner join 을 수행하여 교집합인 결과를 반환.

how 옵션에 'left', 'right', 'outer' 를 설정하여, 왼쪽 조인, 오른쪽 조인, 외부 조인을 수행 할 수 있음
"""

pd.merge(df1, df2, how='outer') #해당 키에 없는 것들은 nan이 반환된다.

pd.merge(df1, df2, how='left')

pd.merge(df1, df2, how='right')

"""여러 개의 키를 병합하여 병합된 결과를 키로 사용하려면 컬럼 이름이 담긴 리스트를 넘기면 된다."""

left = pd.DataFrame({'key1':['foo','foo','bar'],
                     'key2':['one','two','one'],
                     'key3':[1,2,3]})

right = pd.DataFrame({'key1':['foo','foo','bar', 'bar'],
                     'key2':['one','one','one', 'two'],
                     'key3':[4,5,6,7]})

pd.merge(left, right, on=['key1','key2'], how='outer')

"""**index를 key로 사용하기**

병합하려는 키가 Dataframe의 인데스인 경우가 있음. 이런 경우에는

**left_index = True, right_index=True** 옵션을 지정해서 해당 인덱스를 병합키로 사용

인덱스를 병합키로 사용한다! right_index = TRUE 해주면 된다.


"""

left1 = pd.DataFrame({'key':['a','b','a','a','b','c'],
                      'value': range(6)})

right1 = pd.DataFrame({'group_val':[3.5, 7]}, index=['a','b'])

pd.merge(left1, right1, left_on='key', right_index=True)

"""#merge는 기본적으로 inner join을 수행한다.

**B. Join**

**인덱스 병합**의 경우 DataFrame의 join 메서드를 사용

join 메서드는 컬럼이 겹치지 않으며 완전히 같거나 유사한 색인 구조를 가진 여러 개의 DataFrame 객체를 병합할 때 사용

#기억하자! join은 컬럼이 겹치지 않으며!! 같거나 유사한 인덱스를 가진 여러 데이터프레임을 병합한다.
"""

left2 = pd.DataFrame([[1,2],[3,4],[5,6]],
                     index = ['a','c','e'],
                     columns=['Ohio','Nevada'])

right2 = pd.DataFrame([[7,8],[9,10],[11,12],[13,14]],
                      index=['b','c','d','e'],
                      columns=['Missouri','Alabama'])

left2.join(right2, how='outer') #이건 매서드!! so df객체에다가 붙인다.

"""과거에 작성된 pandas 일부 코드 제약으로 DataFrame의 join 메서드는 왼쪽 조인을 수행한다. 

join 메서드를 호출한 DataFrame의 컬럼 중 하나에 대해 조인을 수행하는 것도 가능하다.
"""

print(left1)
print(right1)

left1.join(right1, on='key') #컬럼이 겹쳐지지 않는다는 것에 집중해봐라!!!

"""단지, 인덱스 병합을 할 경우에는 병합하려는 DataFrame의 리스트를 join 메서드로 전달."""

another = pd.DataFrame([[7,8],[9,10],[11,12],[16,17]],
                       index=['a','c','e','f'],
                       columns=['New York', 'Oregon'])

left2.join([right2,another]) #그냥 인덱스 병합할때는 병합하려는 df의 리스트를 전달하자!!!!

"""**C. concat**

axis = 0 인 경우, row 를 추가, 

axis = 1 인 경우, column 추가
"""

s1 = pd.Series([0,1,3], index=['a','b','c'])
s2 = pd.Series([2,3,4], index=['c','d','e'])
s3 = pd.Series([5,6], index=['f','g'])

pd.concat([s1, s2, s3]) # 디폴트 0 즉, 로우

pd.concat([s1, s2, s3], axis=1) #인덱스가 다 다르니까 행이 길어질 수 밖에 없지

"""**join = 'inner'**

join='inner' 옵션을 전달하여, inner join 을 수행할 수 있음
"""

pd.concat([s1, s2],axis=1, join='inner') #concat도 조인이 되는구나

"""**join_axes = [] : 삭제된 옵션**

join_axes 옵션에 병합하려는 축을 직접 지정할 수 있음

**.reindex()**

join_axes 옵션을 전달하는 것과 같은 효과로, 병합하려는 축의 리스트를 인자로 넘겨줌
"""

# pd.concat([s1, s2, s3], axis=1), join_axes=['a','b])
pd.concat([s1, s2, s3], axis=1).reindex(['a','b']) # reindex, just remake index and concat

"""**ignore_index = True**

인덱스가 데이터 분석에 불필요한 경우, ignore_index 를 새로운 인덱스를 생성
"""

df1 = pd.DataFrame(np.random.randn(3,4), columns=['a','b','c','d'])
df2 = pd.DataFrame(np.random.randn(2,3), columns=['b','d','e'])

df2

pd.concat([df1, df2], ignore_index=True) #인덱스 필요없으니까 그냥 붙여!

"""**D. pivot & melt**"""

df = pd.DataFrame({'key':['foo','bar','baz'],
                   'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9]})
df

melted = pd.melt(df, ['key']) #분리하기
melted

reshaped = melted.pivot('key', 'variable', 'value') #아니 그냥 순서가 이래 키, 변수, 값으로 넣어준다.
 #피벗은 약간 별동대 느낌 새로운 조합으로 만들 수 있다. .pivot
reshaped #variable은 변수!

"""#피벗, row의 값들이 column이 된다. so 변수가 될 값은 norminal이어야 겠다.
df.pivot(index='', columns='', key='')
"""

reshaped.reset_index() #그냥 새로운 데이터 객체 만들었다. 인덱스를 초기화한다.

import pandas as pd
import numpy as np

#'ValueError: Index contains duplicate entries, cannot reshape'
#이 말인 즉, 값의 개수가 같아야 한다는 뜻이다. duplicate안된다잖아 따라서 정제 해야한다. 
df2 = pd.DataFrame({'key':['foo','bar','baz'],
                   'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9],
                    'D': [11,12,10]})

a = df2.melt(df2,['key'])
print(a)

df = pd.DataFrame([
        ['20180901', 'A',   10],
        ['20180901', 'B',  100],
        ['20180901', 'C', 1000],
        ['20180902', 'A',   20],
        ['20180902', 'B',  200],
        ['20180902', 'C', 2000],
        ['20180903', 'A',   30],
        ['20180903', 'B',  300],
        ['20180903', 'C', 3000],
], columns=['date', 'typecode', 'volume'])

df_pivoted = df.pivot(index='date', columns='typecode', values='volume' )
df_pivoted.reset_index(inplace = True)
df_pivoted.index
##중요한 점은 듀플리케이트인 로우가 없어야 한다. 이 말인 즉, 인덱스가 date인데 해당 데이트에 A값이 두개이면, 동일 인데스가 생긴다. 이에 오류가 발생함
