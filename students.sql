-- Drop tables if they already exist
DROP TABLE IF EXISTS Examinations;
DROP TABLE IF EXISTS Students;
DROP TABLE IF EXISTS Subjects;

-- Create Students table
CREATE TABLE Students (
    student_id INTEGER PRIMARY KEY,
    student_name VARCHAR(50) NOT NULL
);

-- Create Subjects table
CREATE TABLE Subjects (
    subject_name VARCHAR(50) PRIMARY KEY
);

-- Create Examinations table
CREATE TABLE Examinations (
    student_id INTEGER,
    subject_name VARCHAR(50),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (subject_name) REFERENCES Subjects(subject_name)
);

-- Insert data into Students
INSERT INTO Students (student_id, student_name) VALUES
(1, 'Alice'),
(2, 'Bob'),
(13, 'John'),
(6, 'Alex');

-- Insert data into Subjects
INSERT INTO Subjects (subject_name) VALUES
('Math'),
('Physics'),
('Programming');

-- Insert data into Examinations
INSERT INTO Examinations (student_id, subject_name) VALUES
(1, 'Math'),
(1, 'Physics'),
(1, 'Programming'),
(2, 'Programming'),
(1, 'Physics'),
(1, 'Math'),
(13, 'Math'),
(13, 'Programming'),
(13, 'Physics'),
(2, 'Math'),
(1, 'Math');
