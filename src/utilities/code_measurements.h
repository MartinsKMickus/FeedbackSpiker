

/// <summary>
/// Saves current time that can be used to get elapsed time with stop_chronometer()
/// </summary>
void start_chronometer();

/// <summary>
/// Returns elapsed time since start_chronometer()
/// </summary>
/// <returns>Elapsed time in unit defined with set_chronometer_unit_unitname(), default ms</returns>
double stop_chronometer();

/// <summary>
/// Sets chronometer measurement time unit to seconds
/// </summary>
void set_chronometer_unit_seconds();

/// <summary>
/// Sets chronometer measurement time unit to miliseconds
/// </summary>
void set_chronometer_unit_miliseconds();

/// <summary>
/// Sets chronometer measurement time unit to microseconds
/// </summary>
void set_chronometer_unit_microseconds();