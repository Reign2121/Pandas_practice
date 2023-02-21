
"""
Pandas 시작하기

**import pandas as pd**

1. 고수준의 자료구조(Series, Dataframe)와 파이썬에서 빠르게 쉽게 사용할 수 있는 데이터 분석도구를 포함하고 있음.
2. Numpy, Scipy, Statsmodels, Scikit-learn 등과 함께 사용.
3. **Pandas의 핵심은 for 문을 사용하지 않고 데이터를 처리할 수 있다는 것이며,  배열기반의 함수를 제공한다는 것!**
4. Numpy의 배열과 유사하나, Numpy는 단일 산술 배열 데이터를 다루는데 특화되어 있는 반면, pandas 는 표 형식의 데이터나 다양한 형태의 데이터를 다루는 데 초점을 맞춰 설계했다는 것임

판다스 모듈을 임포트 한다.
pd 는 namespace 이름으로, 다른 것으로 칭하여도 무방하나, pd가 일반적임.
"""

import pandas as pd
import numpy as np

"""Pandas 의 자료구조 소개(1)

**1. Series**

A. list, tuple, np.array 등의 iterable 한 객체로 생성

B. 1차원 배열과 같은 구조이며, index 라고 하는 배열의 데이터와 연관된 값을 가지고 있음. Python의 1차원 dictionary와 유사

C. 값을 확인하기위한 .value

D. 인덱스를 확인하기 위한  .index
"""

#list
_li = [4,7,-5,3]
#tuple
_tp = (4,7,-5,3)

li_sr = pd.Series(_li)
print(li_sr)
np_sr = pd.Series(np.array(_li))
print(type(np_sr))
tp_sr = pd.Series(_tp)
print(tp_sr)
type(tp_sr)

_li = [4, 7, -5, 3]
_index = ['a','b','c','d'] #위치를, 오브젝트인 인덱스를 가지고 있다. 
_dict = {'e':4,'f':7, 'g':-5, 'h':3}
li_sr = pd.Series(_li, index=_index)
dict_sr = pd.Series(_dict)
print(li_sr)
print(li_sr.values)
print(li_sr.index)
print(dict_sr)

"""E. Numpy와 다른 점은 1) 단일값혹은 여러값을 선택할 때 인덱스로 라벨을 사용할 수 있다.

---



F. Numpy와 다른 점은  2) **불리언 배열**을 사용해서 값을 걸러내거나 산술 곱셈을 수행할 수 있다. ( map, apply, filter 의 기능을 수행 ) 
"""

print(li_sr['a'])
print(li_sr[['a','b','c']])
print(li_sr*2)
print(li_sr[li_sr>2])

_ar = np.array(li_sr)
_ar = filter(lambda x: x>2, _ar)
print(_ar)
_ar = np.array(list(_ar))
print(_ar)

"""G. 인덱스를 바꾸는 경우, 

**.reindex(_index)** 을 사용

1) 타겟 Series에 인덱스가 없다면, 인덱스를 생성하고, NaN 값을 value에 채우고,

2) 인덱스가 존재한다면, 새로 제공되는 인덱스의 순서로 정렬

3) 새로 제공되는 인덱스에 타겟 Series 의 인덱스가 없다면, 해당 index와 value를 제거
"""

sData = {'a':2, 'c':4, 'b':6}
sr = pd.Series(sData)
print(sr)
sr = sr.reindex(['b','c','d'])
print(sr)

"""F. 누락된 데이터를 찾는 경우,

**.isnull(), .notNull()**

불리언 인덱싱을 통해 filter 를 수행할 수 있음
"""

print(sr.isnull())
print(sr.notnull())
print(sr[sr.notnull()])

"""G. Sereis 간의 연산을 하는 경우,

A) 같은 인덱스를 가진 값들끼리 연산을 수행하며,

B) 서로 중복되지 않는 인덱스가 존재하는 경우, 인덱스를 생성하고 값을 NaN으로 채움(outer join과 같은 수행)
"""

left_dict = {'a':1,'b':1,'c':1}
left_sr = pd.Series(left_dict)

right_dict = {'b':1,'c':1,'d':1}
right_sr = pd.Series(right_dict)

print(left_sr)
print(right_sr)
print(left_sr+right_sr)

"""F. Series, index 의 labeling

**.name = '   '**

**.index.name = '   '**
"""

print(left_sr)

left_sr.name = 'left'
left_sr.index.name = 'left_index'

print(left_sr)

"""판다스의 자료 구조소개(2)

**2. DataFrame**

엑셀의 스프레드시트 같은 자료구조의 데이터로서,

A. 인덱스를 공유하는 복수의 Series가 함께 있는 형태 

B.
. 리스트를 배열로 갖는 dictionary의 형태

C. 2차원 배열에 인덱스와 컬럼명을 가지고 있는 형태
"""

_li_1 = [1,2,3,4]
_li_2 = [5,6,7,8]
_li_3 = [9,10,11,12]
data = {'column_1':_li_1,'column_2':_li_2, 'column_3':_li_3}
_df = pd.DataFrame(data)
print(_df)


_sr_1 = pd.Series(_li_1)
_sr_2 = pd.Series(_li_2)
_sr_3 = pd.Series(_li_3)

_df_2 = pd.DataFrame()
_df_2['coulumn_1'] = _sr_1
_df_2['coulumn_2'] = _sr_2
_df_2['coulumn_3'] = _sr_3
print(_df_2)

print(np.arange(1, 13))
 
_df_3 = pd.DataFrame(np.arange(12).reshape(4, 3), columns=['column_1', 'column_2', 'column_3'])
print(_df_3)
_df_3.index = ['a','b','c','d']
print(_df_3)

"""D. Dataframe 에서 row와 column 의 선택

1. Column : 인덱싱을 통해 선택 

  **df['column_name']**

2. row : .loc(), .iloc()을 통해 선택 
  
  **.loc['index_name']**, 
  
  **.iloc['index']**
"""

print(_df_3)
print(_df_3['column_1'])
print(_df_3.loc['a'])
print(_df_3.iloc[0])

"""E. Column 의 생성

**_df['column_name'] = data** 를 수행하면,

column_name 이 있는 경우, update ( update 시 인덱스의 개수와 데이터의 개수가 맞지 않으면 Error 발생 )

column_name 이 없는 경우, 새로이 생성
"""

print(_df_3)
data = [3,3,3,3]

_df_3['column_4'] = data

_df_3['column_3'] = data
print(_df_3)

_df_3['column_5'] = _df_3['column_3'] == 3
print(_df_3)

"""F. Column, Row 의 삭제

1. del 예약어를 이용해서 column 삭제

2. .drop() 을 사용하여 column 또는 row 를 삭제 

    ***axis 정보를 parameter로 전달해야 함:*** 
    
    axix=0 -> row 삭제, axis=1 -> column 삭제, default 값은 0

    ***inplace 정보를 parameter로 전달하여, 데이터프레임 자체를 변경 가능:***

    inplace=True -> 해당 데이터프레임 변경, inplace=False -> 변경된 새로운 데이터 프레임 반환. default 값은 False

"""

df = pd.DataFrame(np.arange(12).reshape(3,4), index=['a','b','c'], columns=['c1','c2','c3','c4'])
print(df)

del df['c2']
print(df)

df.drop('c1', axis=1, inplace=True)
print(df)

df.drop('a', axis=0, inplace=True)
print(df)

"""G. DataFrame의 전치

**df.T** 를 통해 axis0 과 axis1을 전치할 수 있음
"""

print(df)
print(df.T)

"""H. 재색인

.reindex([_li_new_index])를 통해 재색인이 가능

기존 인덱스와 _li_new_index 가 중복되는 경우, 순서를 새롭게 정렬하고,

중복되지 않는 경우, 인덱스를 새로이 생성하고 NaN 값으로 채움


G. Interpolation

**method**

.reindex()를 통해 재색인 하는 경우, 데이터가 새롭게 생성되며, NaN 값으로 채워지는 것을 방지하기 위해 **method** 를 파라미터로 입력해주면, NaN 값으 채울 수 있다.

**ffill** : column 의 이전 값으로 NaN을 변경

**bfill** : column 의 이후 값으로 NAM을 변경

**fill_value**

.reindex() 수행시 fill_value 파라미터를 넘겨주면, 특정 값으로 NaN을 대체할 수 있다.
"""

df = pd.DataFrame(np.arange(12).reshape(3,4), index=['a','b','c'])
print(df)
_li = ['b','c','e','a']
print(df.reindex(_li))

print(df.reindex(_li, method='ffill'))

_li_2 = ['g','b','a']
print(df.reindex(_li_2))
print(df.reindex(_li_2, fill_value=0))

"""I. Indexing

**Column 선택하기 **

df['column_name'] 으로 해당 df의 column 을 선택가능

**Row 선택하기**

df[start:end+1]의 slicing 을 통해 row 선택가능

**Column, Row 선택하기** 

1. df.loc[row_name_slicing, column_name_slicing]

2. df.loc[[row_list],[column_list]]

3. df.iloc[row_index_number_slicing, column_index_number_slicing]

4. df.iloc[[row_list],[column_list]]
"""

df = pd.DataFrame(np.arange(12).reshape(3,4), index=['1_row','2_row','3_row'], columns=['1_column','2_column','3_column','4_column'])
print(df)

print(df['1_column'])
print(df[0:2])
print(df.iloc[:,:])
print(df.iloc[0, :])
print(df.iloc[:,0])
print(df.loc['1_row':'3_row', '1_column':'2_column'])
print(df.loc[['1_row', '3_row']])
print(df.iloc[[0,2]])

"""J. 데이터프레임간의 산술연산

데이터 프레임은 **그 자체로 operand**의 역할을 할 수 있음 

단, 데이터프레임이 operand가 되는 경우, outer join 의 형태로 모든 column과 index 가 생성되고, **중복되지 않는 column 과 index 는 NaN으로 채워진다.**

sorting 이후 실행되기 때문에 row와 column 의 순서가 다를 경우, sorting 된 결과가 보여짐
"""

df_left = pd.DataFrame(np.ones((3,3), dtype=int), index=['a','b','c'], columns=['col1','col2','col3'])
print(df_left)
df_right = pd.DataFrame(np.full((4,5), 2), index=['a','c','e','d'], columns=['col2', 'col1', 'col4','col5','col6'])
print(df_right)
print(df_left + df_right)

"""K. 함수 Mapping

**.apply(lambda x: ~ )**: x 인자에 column 을 넘겨 받아 column 단위의 함수매핑을 수행한다.

**.applymap(lambda x: ~ )**: x 인자에 데이터프레임 내의 개별적인 value를 넘겨 받아 value 단위의 함수매핑을 수행한다.

"""

df = pd.DataFrame(np.arange(12).reshape(3,4), index=['a','b','c'], columns=['col_1','col_2','col_3','col_4'])
print(df)

f = lambda x : x.min()

print(df.apply(f))

f = lambda x : x^2

print(df.applymap(f))

"""L. Sort

.sort_index() : 인덱스 혹은 column 명을 기준으로 정렬한다.

  parameter 

  1) axis : 0 인 경우 Row, 1인 경우 Column 기준으로 정렬

  2) ascending : True 인 경우 오름차순, False 인 경우 내림차순 정렬

.sort_values() : 값을 기준으로 정렬 

  parameter 

  1) row 명, column 명

  2) axis : 0 인 경우 Row, 1인 경우 Column 기준으로 정렬

  3) ascending : True 인 경우 오름차순, False 인 경우 내림차순 정렬
"""

_df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['column_1', 'column_2', 'column_3', 'column_4'])

print(_df)
print(_df.sort_index(axis=0, ascending=False))#여기서 axis가 1이면 컬럼의 이름을 기준으로 쓰게 된다.
print(_df.sort_values(1, ascending=False, axis=1))#어센딩도 기본적으로 오름차순이 디폴트.

"""M. 기술 통계 계산과 요약

**default 는 axis = 0 이며, column에 대한 통계**를 보여줌.
axis = 1 인 경우, Row 에 대한 통계를 제공

mean(): 평균

sum(): 합산

idxmax() : max 값의 index

cumsumn() 누산

var, std, skew, kurt 등 의 다양한 통계값을 제공
"""

_df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['column_1', 'column_2', 'column_3', 'column_4'])

print(_df.sum())
print(_df.sum(axis=1))

"""N. Covariance & Correlation

Column 간의 공분산을 및 상관계수를 구해줌

df.cov() : 공분산

df.corr() : 상관계수
"""

_df = pd.DataFrame(np.arange(12).reshape(4,3), columns=['AAPL', 'AMZN', 'TESLA'])
print(_df)
print(_df.cov())
print(_df.corr())