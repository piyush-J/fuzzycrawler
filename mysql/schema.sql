CREATE DATABASE fuzzycrawler_test;
USE fuzzycrawler_test;
CREATE TABLE `user`
(
  `id` bigint NOT NULL,
  `fname` varchar
(255) NOT NULL,
  `lname` varchar
(255) NOT NULL,
  `email` varchar
(400) NOT NULL,
  `pass` varchar
(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

ALTER TABLE `user`
ADD PRIMARY KEY
(`id`),
ADD UNIQUE KEY `email`
(`email`);
ALTER TABLE `user`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT;
COMMIT;

INSERT INTO `user`
  (`fname`,`lname`
  ,`email`,`pass`) VALUES
('John','Doe','john.doe@example.com','Doe@123');
