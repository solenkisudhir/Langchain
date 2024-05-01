# Streaming | 🦜️🔗 LangChain
All ChatModels implement the Runnable interface, which comes with default implementations of all methods, ie. ainvoke, batch, abatch, stream, astream. This gives all ChatModels basic support for streaming.

Streaming support defaults to returning an Iterator (or AsyncIterator in the case of async streaming) of a single value, the final result returned by the underlying ChatModel provider. This obviously doesn’t give you token-by-token streaming, which requires native support from the ChatModel provider, but ensures your code that expects an iterator of tokens can work for any of our ChatModel integrations.

```
 Here's a song I just improvised about goldfish on the moon:

Floating in space, looking for a place 
To call their home, all alone
Swimming through stars, these goldfish from Mars
Left their fishbowl behind, a new life to find
On the moon, where the craters loom
Searching for food, maybe some lunar food
Out of their depth, close to death
How they wish, for just one small fish
To join them up here, their future unclear
On the moon, where the Earth looms
Dreaming of home, filled with foam
Their bodies adapt, continuing to last 
On the moon, where they learn to swoon
Over cheese that astronauts tease
As they stare back at Earth, the planet of birth
These goldfish out of water, swim on and on
Lunar pioneers, conquering their fears
On the moon, where they happily swoon

```
