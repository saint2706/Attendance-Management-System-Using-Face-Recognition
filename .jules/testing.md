To optimize test execution speed, replace iterative usage of `User.objects.create_user()` with a pre-hashed password using `make_password()` and `User.objects.create()`
