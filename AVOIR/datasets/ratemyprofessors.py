import pandas as pd
import numpy as np
from os import path
from .dataset import Dataset
from collections import Counter
from nltk import word_tokenize

DATA_DIR = path.join(path.dirname(__file__), "data", "ratemyprofessors")
RAW_DATA_FILE = "raw_ratings.csv"
PROCESSED_DATA_FILE = "ratings.csv"

TRAIN_PORTION = 0.8
DATA_SIZE = None

original_attributes = {
    'comments': 'Comments left by students on rating',
    'department_name': 'Department name of course',
    'state_name': "US State",
    'year_since_first_review': "Teaching age, from the first student evaluation to the time of analysis in year 2019.",
    'star_rating': 'Aggregate Rating, 3.5-5.0 is good, 2.5-3.4 is average and 1.0-2.4 is poor',
    'take_again': "Percentage, e.g. '89%'",
    'diff_index': "Aggregate Difficulty. 1 Easiest, 5 Hardest",
    'student_star': 'Student Rating, 3.5-5.0 is good, 2.5-3.4 is average and 1.0-2.4 is poor',
    'student_difficult': "Student Difficulty. 1 Easiest, 5 Hardest",
    'attence': "Attendance Policy: Mandatory or not",
    'for_credits': "Does the course count for credits",
    'would_take_agains': "Student indication of if they would take the course again",
    'grades': 'Grade Received',
    'word_comment': 'Word count of student comment',
    'gender': 'Student Gender',
    'race': 'Student Race',
    'asian': 'Percentage of asian students',
    'hispanic': 'Percentage of hispanic students',
    'nh_black': 'Percentage of black students',
    'nh_white': 'Percentage of white students',
    'gives_good_feedback': 'Does the Professor give good feeback on average',
    'caring': 'Is the professor caring on average',
    'respected': 'Is the professor respected on average',
    'participation_matters': 'Is it perceived that participation matters on average',
    'clear_grading_criteria': 'Is it perceived that grading is clear on average',
    'skip_class': 'Is it perceived that you can skip class on average',
    'amazing_lectures': 'Is it perceived that lectures are amazing on average',
    'inspirational': 'Is it perceived that the professor is inspirational on average',
    'tough_grader': 'Is it perceived that the professor is a tough grader on average',
    'hilarious': 'Is it perceived that the professor is hilarious on average',
    'get_ready_to_read': 'Do students feel you need to read on average',
    'lots_of_homework': 'Do students feel there is lots of homework on average',
    'accessible_outside_class': 'Do students feel that the professor is accessible outside class on average',
    'lecture_heavy': 'Do students feel that the professor is lecture heavy on average',
    'extra_credit': 'Does the professor offer extra credit?',
    'graded_by_few_things': 'Is the grade dependent on few assignments',
    'group_projects': 'Are group projects common',
    'test_heavy': 'Is the class test heavy',
    'so_many_papers': 'Do students feel that the course has a lot of paper assignments',
    'beware_of_pop_quizzes': 'Does the professor give pop quizzes',
    'IsCourseOnline': 'Is the course offered online'
}

attributes = ['comments', 'year_since_first_review', 'star_rating', 'take_again',
              'diff_index', 'student_star', 'student_difficult',
              'for_credits', 'would_take_agains', 'grades',
              'word_comment', 'asian', 'hispanic',
              'nh_black', 'nh_white', 'gives_good_feedback',
              'caring', 'respected', 'participation_matters',
              'clear_grading_criteria', 'skip_class', 'amazing_lectures',
              'inspirational', 'tough_grader', 'hilarious',
              'get_ready_to_read', 'lots_of_homework', 'accessible_outside_class',
              'lecture_heavy', 'extra_credit', 'graded_by_few_things',
              'group_projects', 'test_heavy', 'so_many_papers',
              'beware_of_pop_quizzes', 'IsCourseOnline', 'male_dominated_department',
              'female_dominated_department', 'undominated_department', 'state_name',
              'attence', 'gender',
              'student_race']

attributes_dict = dict((attr, "") for attr in attributes)

for attr, description in original_attributes.items():
    attributes_dict.update({
        matched_attr: description
        for matched_attr in attributes
        if matched_attr.startswith(attr)
    })

# TODO create attribute for male dominated department, female, and other
department_map = {
    'male_dominated': ['Astronomy department', 'Computer Science department', 'Physics department',
                       'Chemistry department', 'Philosophy department', 'Science department',
                       'Engineering department', 'Electrical Engineering department', 'Geography department',
                       'Geology department', 'Theology department', 'Religion department',
                       'Chemistry & Biochemistry department', 'Aviation department', 'Religious Studies department',
                       'Computer Information Systems department', 'Civil Engineering department', 'Computer Engineering department',
                       'Automotive Technology department', 'Electrical Technology department', 'Materials Science department',
                       'MacRomolecular Science & Eng department', 'Natural Sciences department', 'Mechanical Engineering department'],
    'female_dominated': ['English department', 'Writing department', 'ASL & Deaf Studies department',
                         'Education department', 'Communication department', 'Languages department',
                         "Women\\'s Studies department", 'Theater department', 'Fine Arts department',
                         'Psychology department', 'Graphic Arts department', 'Journalism department'
                         'Health Science department', 'Humanities department', 'Childrens Literature department',
                         'Visual Arts department', 'Management department', 'Law department',
                         'Linguistics department', 'Medicine department', 'Literature department',
                         'Speech department', 'Ethnic Studies department', 'Asian American Studies department',
                         'Anatomy department', 'Pharmacology department', 'African Studies department',
                         'Elementary Education department', 'Nursing department', 'Art department',
                         'Family & Child Studies department', 'German department', 'Pharmacy department',
                         'Family & Consumer Science department', 'Comparative Literature department', 'Hispanic Studies department',
                         'Spanish department', 'Interaction Design & Art department', 'Public Health department'],
    'other': ['Architecture department', 'History department', 'Mathematics department',
              'Economics department', 'Biology department', 'Political Science department',
              'Business department', 'Sociology department', 'International Studies department'
              'Social Science department', 'Anthropology department', 'Accounting department',
              'Not Specified department', 'Design department', 'Finance department',
              'Culinary Arts department', 'Art History department', 'Music department',
              'Physical Ed department', 'Film department', 'Italian department',
              'Criminal Justice department', 'Marketing department', 'Library Science department',
              'Hospitality department', 'Agriculture department', 'Biochemistry department',
              'Nutrition department', 'Kinesiology department', 'Classics department',
              'Statistics department', 'Accounting & Finance department', 'Chicano Studies department',
              'Environment department', 'Russian department', 'Earth Science department',
              'Honors department', 'Physical Education department']
}


def _extract_category_from_department(group_name, include_unknowns=False):
    def f(dep_name):
        if dep_name not in department_map[group_name]:
            return 1 if include_unknowns else 0
        return 1
    return f


def _letter_to_gpa(letter: str) -> float:
    gpa_map = {
        'A+': 4,
        'A': 4,
        'A-': 3.7,
        'B+': 3.3,
        'B': 3,
        'B-': 2.7,
        'C+': 2.3,
        'C': 2,
        'C-': 1.7,
        'D+': 1.3,
        'D': 1,
        'D-': 0.7,
        'F': 0
    }
    return gpa_map[letter] if letter in gpa_map else np.nan


def _percent_str_to_float(s):
    if pd.isnull(s):
        return np.nan
    return float(s.strip('%'))/100


def _yes_no_to_binary(col):
    def f(x):
        if pd.isnull(x):
            return np.nan
        return 1 if x == 'Yes' else 0
    return col.apply(f)


def _get_gender(df):
    data = df.copy()
    data.comments = data.comments.apply(lambda c: "No Comment" if pd.isnull(c) else c)

    grouped = data.groupby(by=["professor_name", "department_name", "school_name"], as_index=False).agg({
        "gender": np.max,
        "comments": lambda x: ' '.join(x)
    })

    female_pronouns = {"she", "her", "hers", "mrs", "herself", "woman"}
    male_pronouns = {"he", "his", "him", "mr", "mister", "himself", "man"}

    def gender_classify(row):
        comments = row.comments.lower()
        comments = comments.replace("\\", "")
        word_count = Counter(word_tokenize(comments))

        # count all male and female pronouns used in the comments
        female_words = np.sum([word_count[x] for x in female_pronouns])
        male_words = np.sum([word_count[x] for x in male_pronouns])

        if female_words > male_words:
            return "female"
        elif male_words > female_words:
            return "male"
        #elif "female" in row.gender:
        #    return "female"
        #elif "male" in row.gender:
        #    return "male"
        else:
            return "unknown"

    grouped.gender = grouped.apply(lambda row: gender_classify(row), axis = 1)

    def fetch_gender(row):
        return grouped[grouped.professor_name == row.professor_name].iloc[0].gender

    return data.apply(fetch_gender, axis=1)

    
def _raw_data():
    data_fp = path.join(DATA_DIR, RAW_DATA_FILE)
    data_raw = pd.read_csv(data_fp)

    return data_raw

def _process_raw_data():
    data_fp = path.join(DATA_DIR, RAW_DATA_FILE)
    data_raw = pd.read_csv(data_fp)

    # subset only relevant attributes
    data = data_raw[list(original_attributes.keys())].copy()

    # convert percent strings to floats
    data.take_again = data.take_again.apply(_percent_str_to_float)

    # convert yes no columns to binary
    data.for_credits = _yes_no_to_binary(data.for_credits)
    data.would_take_agains = _yes_no_to_binary(data.would_take_agains)

    # convert letter grades to gpa
    data.grades = data.grades.apply(_letter_to_gpa)

    # get gender based on comments
    data.gender = _get_gender(data_raw)

    # categorize departments
    data['male_dominated_department'] = data.department_name.apply(
        _extract_category_from_department("male_dominated"))
    data['female_dominated_department'] = data.department_name.apply(
        _extract_category_from_department("female_dominated"))
    data['undominated_department'] = data.department_name.apply(
        _extract_category_from_department("other", include_unknowns=True))
    data.drop('department_name', axis=1, inplace=True)

    # impute na values
    data.take_again.fillna(data.take_again.mean(), inplace=True)
    data.student_star.fillna(data.student_star.mean(), inplace=True)
    data.student_difficult.fillna(data.student_difficult.mean(), inplace=True)
    data.for_credits.fillna(int(data.for_credits.mean()), inplace=True)
    data.would_take_agains.fillna(
        int(data.would_take_agains.mean()), inplace=True)
    data.grades.fillna(data.grades.mean(), inplace=True)
    data.word_comment.fillna(0, inplace=True)
    comments = data.comments.fillna(" ").copy()
    comments = comments.replace("\\\\", "")
    # dummy_columns = [c for c in data.columns if c != 'comments']
    # data = pd.get_dummies(data, columns=dummy_columns)
    data = data.drop(["comments"], axis=1) 
    data["comments"] = comments
    
    
    ## add race as 'student_race'
    data["student_race"] = data.race.copy()
    data = data.drop(["race"], axis=1)
    # to convert categorical to binary
    #data = pd.get_dummies(data)
    return data

def _read_data():
    processed_data_fp = path.join(DATA_DIR, PROCESSED_DATA_FILE)

    try:
        data = pd.read_csv(processed_data_fp)
    except FileNotFoundError:
        data = _process_raw_data()
        data.to_csv(processed_data_fp, index=False)


    data_size = DATA_SIZE if DATA_SIZE is not None else len(data)
    train_size = int(data_size * TRAIN_PORTION)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:data_size]

    return train, test


def load_data() -> Dataset:
    train, test = _read_data()

    return Dataset(
        train=train,
        test=test,
        attributes_dict=attributes_dict,
        target="student_star"
    )
