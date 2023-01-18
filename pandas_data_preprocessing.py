# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

"""#**데이터 정제 및 준비**

**A. 누락된 데이터 처리하기**

1. pandas의 설계 목표 중 하나는 누락 데이터를 가능한 한 쉽게 처리하게 하기 위함.
2. 1로 인해 pandas 객체의 모든 기술 통계는 누락된 데이터를 배제하고 처리. null제외 연산 다 된다.
3. pandas는 누락된 데이터를 NaN으로 취급
4. 누락데이터를 찾는 **.isnull()**

#**NaN** 

pandas에서는 R프로그래밍 언어에서 결측치를 NA로 취급하는 개념을 차용했음
1. python의 내장 None
2. numpy의 np.nan
"""

string_data = pd.Series(['aardvark','artichoke',np.nan,'avocado'])
print(string_data)

string_data.isnull()

string_data[0] = None
string_data.isnull()

#결측치를 나타내는 이름이 좀 특이하다. 파이썬 내장에서는 none이라고 부르는구나,, numpy에서는 nan이라고 부른다. pandas도 nan개념

"""**B. 누락된 데이터 골라내기**

**pandas.dropna()**

pandas.isnull()이 불리언 색인을 반환하는 반면, pandas.dropna()의 경우 non-null 인 데이터와 색인값만 들어 있는 Series를 반환
"""

data = pd.Series([1, np.nan, 3.5, np.nan, 7])
data.dropna()

"""흔히 사용하는 code로서는 위와 동일한 코드는 다음과 같다."""

data[data.notnull()]

"""**DataFrame의 경우,**

1. DataFrame 객체의 경우에는 모두 NaN 값인 로우나 컬럼을 제외시키거나,
2. NaN 값을 하나라도 포함하고 있는 로우나 컬럼을 제외시킬 수 있음.
3. pandas.dropna()는 기본적으로 NaN값을 하나라도 포함하고 있는 로우를 제외시킴 (파괴적이다)
"""

data = pd.DataFrame([[1, 6.5, 3],[1, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3]])
cleaned = data.dropna()
print(data)
print(cleaned)

"""**how = 'all'** 

how='all' 을 pandas.dropna()의 parameter로 전달하면, row 의 모든 값이 NaN인 로우만 제외시킴 
"""

data.dropna(how='all')
#디폴트가 다 지우는거니까 all하면 된다.

"""**axis = 1**

dropna 는 기본적으로 row를 기준으로 작동하나, axis = 1로 옵션값을 주면(parameter로 넘겨주면), column을 기준으로 작동함
"""

print(data.dropna())

print(data.dropna(axis=1))

data[4] = np.nan

print(data)

print(data.dropna(axis=1))

print(data.dropna(axis=1, how='all'))

"""**thresh = n**

n개 이상의 NaN값이 있는 Row를 제외시키고 싶은 경우 thresh 옵션을 전달

"""

df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan

print(df)
print(df.dropna())
print(df.dropna(thresh=2)) #all대신 개수를 지정해줄 수 있다

"""**C. 결측치 채우기**

**pandas.fillna()**

누락된 값을 제외시키는 것은 좋은 방법이 아님 -> 잠재적으로 버려지지 않아야 할 데이터가 버려져 데이터가 왜곡될 수 있음.

데이터 상의 '구멍'을 어떻게든 메우고 싶은 경우가 있는데. 이 경우 pandas.fillna()를 사용
#그래도 데이터에 어떠한 변형을 가하는 것은 지양해야 한다.

"""

print(df)

df.fillna(0)

"""pandas.fillna()에 dictionary 형태의 예약값을 넘겨서 각 컬럼마다 다른 값을 채울 수도 있음."""

df.fillna({1:0.5, 2:0}) ## {컬럼명 : 채울 값} 이렇게 해준다

"""inplace = True

fillna 는 새로운 객체를 반환하지만 inplace=True 옵션을 넘겨주어 기존 객체를 변경할 수도 있음. 
"""

df = df.fillna(0, inplace=True)

"""**method, limit**

method는 보간방법을 결정해 주며, column의 이전 값을 받아 채우는 ffill 과 뒷 값을 받아 채우는 bfill 이 있음.

limit 은 모든 값을 같은 값으로 채우는 것을 막기 위해, 채울 값의 개수를 한정하는데 사용

"""

df = pd.DataFrame(np.random.randn(6,3))
df.iloc[2:, 1] = np.nan
df.iloc[4:, 2] = np.nan
print(df)
print(df.fillna(method='ffill'))

print(df.fillna(method='ffill', limit=2)) #limit은 채울 값의 수


"""
**통계적 수치를 통한 보간**

#fillna()의 값에 mean()과 같은 통계적 수치를 전달하여, 조금 더 세밀한 접근이 가능
#통계적 수치는 column 을 기준으로 계산됨. (자동으로)
"""

df.fillna(df.mean())

df.fillna(df.max())

"""**D. 중복 제거하기**

실제 데이터에서는 여러 가지 이유로 DataFrame에서 중복된 로우를 발견할 수 있다.
"""

data = pd.DataFrame({'k1':['one','two']*3 + ['two'], 'k2':[1,1,2,3,3,4,4]})
data

"""**pandas.duplicated()**

DataFrame의 duplicated 메서드는 각 로우가 중복인지 아닌지 알려주는 불리언 Series를 반환
"""

# data.duplicated()
data.duplicated() #중복인지 아닌지 알려주는 불리언 시리즈 반환

"""**pandas.drop_duplicates()**
drop_duplicates 는 duplicated 배열이 False인 DataFrame을 반환
"""

data.drop_duplicates() #중복인 로우들 제거한, 매서드 결과 true인 로우들 제거 후 반환

"""이 두 메서드는 기본적으로 모든 컬럼에 적용되며 중복을 찾아내기 위해 각 컬럼에 적용이 가능"""

data.drop_duplicates(['k1']) #알다시피 칼럼을 지정할떄는 ['칼럼 명'] 이렇게 지정한다.

"""**keep='last'**

최근 데이터를 살리고 싶은 경우가 있으며, 이러한 경우 keep='last' 옵션을 전달

"""

data.drop_duplicates(keep='last') #최근 데이터 빼고 지워라라는 파라미터를 제공한다.

"""**E. 함수나 매핑을 이용해서 데이터 변형하기**

데이터를 다루다 보면 DataFrame의 컬럼이나 Series, 배열 내의 값을 기반으로 데이터의 형태를 변환하고 싶은 경우가 있으며, 이러한 경우 map 함수를 사용
"""

import pandas as pd
import numpy as np
data = pd.DataFrame({'food':['bacon', 'pulled pork','bacon','Pastrami','corned beef','Bacon','pastrami','honey ham','nova lox'],
                      'ounces':[4,3,12,6,7.5,8,3,5,6]})
data

"""해당 육류가 어떤 동물의 고기인지 알려 줄수 있는 컬럼을 하나 추가한다고 가정하고 육류별 동물을 담고 있는 사전 데이터를 아래처럼 작성"""

meat_to_animal = {'bacon':'pig',
                   'pulled pork':'pig',
                   'pastrami':'cow',
                   'corned beef':'cow',
                   'honey ham':'pig',
                   'nova lox':'salmon'}

"""**str.lower()**

meat_to_animal의 모든 key 값이 소문자로 되어 있으나, data의 'food'컬럼의 value에 대문자가 있으므로, data 의 'food'컬럼을 소문자로 변경
"""

lowercased=data['food'].str.lower()

"""**pandas.map()**

lowercased 를 기준으로 meat_to_animal을 적용시켜, 새로운 animal 컬럼을 생성
"""

data['animal'] = lowercased.map(meat_to_animal) #map은 말그대로 매핑해준다는 의미이다. 연결시켜준다.!!
data

"""**F. 값 치환하기**

데이터를 다루다 보면, np.inf, np.nan, -999.0 등의 값을 확실하지 않은 값의 플래그로 데이터를 채워놓은 경우가 있음.

**pandas.replace()**

이러한 경우 replace()메서드를 통해 일괄적으로 값을 변경할 수 있음.
"""

data = pd.Series([1,-999, 2, -999, -1000, 3])
data

data.replace(-999, np.nan)

"""여러 개의 값을 한 번에 치환하려면 하나의 값 대신 치환하려는 값의 리스트를 넘기면 된다. (여러 개 원하면 리스트를 넘겨라)"""

data.replace([-999, -1000], np.nan)

"""치환하려는 값마다 다른 값으로 치환하려면 누락된 값 대신 새로 지정할 값의 리스트를 사용"""

data.replace([-999, -1000], [np.nan, 0])

"""**G. 개별화와 양자화**

개별로 나누거나, 분석을 위해 그룹별로 나누는 경우
"""

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

"""**pandas.cut()**

위 데이터를 pandas의 cut 함수를 이용해서 18-25, 26-35, 35-60, 60+ 그룹으로 나누는 경우

1. 기준점으로 사용될 지점들의 리스트 생성
"""

bins= [18, 25, 35, 60, 100]

"""2. pandas.cut()메서드를 이용해, 해당 데이터에 기준점 적용 

#pd.cut(대상, 기준점)
"""

cats = pd.cut(ages, bins) #이미 나눈 구간을 전달한다.

type(cats)
cats #카테고리컬 이라는 새로운 객체형 기억하자!!

"""pandas 에서 반환하는 객체는 Categorical 이라는 특수한 객체임.

**.codes** 
몇 번째 그룹에 속해 있는지를 반환
"""

cats.codes #cats 내의 원소가! 어느 그룹에 속해있는지를 반환한다!

"""**.value_counts()**

pandas.cut() 결과에 대한 그룹 수의 통계치
"""

cats.value_counts()

"""**labels**

category 컬럼에 적합한 그룹의 이름을 직접 넘겨 주어 수행할 수 있음 
"""

group_names = ['Youth','YoungAdult','MiddleAged','Senior']

cats = pd.cut(ages, bins, labels=group_names) #이름까지 넘겨준다

list(cats)

"""**균등 분할**

명시적으로 경계값을 넘기지 않고 그룹의 개수를 넘겨주면 데이터에서 최솟값과 최댓값을 기준으로 균등한 길이의 그룹을 자동으로 계산하여 반환

"""

data = np.random.rand(20)

pd.cut(data, 4) #bins를 넘겨주지 않으면 자동으로 균등길이 계산해준다!

"""**qcut**

분위점을 기준으로 데이터를 분류해주거나, 변위치를 직접 지정하여 분류
"""

data = np.random.randn(1000)
cats = pd.qcut(data, 4)
cats
cats.value_counts()

cats = pd.qcut(data, [0,0.1,0.5,0.9, 1])
cats.value_counts()

"""**H. 특잇값을 찾고 제외하기**

배열 연산을 수행할 때는 특잇값(outlier)를 찾아 제외하거나 적당한 값으로 대체하는 것이 중요
"""

data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()

"""이 DataFrame의 한 컬럼에서 절댓값이 3을 초과하는 값 찾기"""

col = data[2]
col[np.abs(col)>3] #np.abs = 절댓값 반환!
data[2][np.abs(data[2])>3] #코드가 너무 더러우니 위의 코드처럼 최적화 하세요.

"""해당 조건을 만족하는 값이 들어 있는 모든 로우를 선택하려면 Boolean DataFrame에서 any 메서드를 사용"""

data[(np.abs(data)>3).any(1)] #any() any(값) 값이 들어있는 모든 로우를 가져와라
#근데 여기서 값은 불리언임. so 1을 의미하는 true를 반환할 것임

"""**I. 치환과 임의 샘플링**


"""

df = pd.DataFrame(np.arange(20).reshape((5,4)))
df

df.sample(n=2, axis = 1) #디폴트 0(행)

"""복원추출을 통해 새로운 데이터셋을 생성하려면"""

choices = pd.Series([5, 7, -1, 6, 4])
draw = choices.sample(n=10, replace=True) # .sample 매서드!! (n 개수와 복원여부 = T or F)
draw
