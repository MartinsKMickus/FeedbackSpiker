
include(CTest) # Include testing capabilities
add_executable(FeedbackSpikerTest
    tests_main.c
    ../src/utilities/text_formatter.c
    ../src/spiker_network.c
)
# target_link_libraries(tests PRIVATE your_library)


add_test(NAME NetworkInitializationTest COMMAND FeedbackSpikerTest network_initialization)
add_test(NAME NeuronAddTest COMMAND FeedbackSpikerTest add_neuron)