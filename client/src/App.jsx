import React, {Component} from 'react';

import Auth from 'pages/Auth';
import Layout from 'components/Layout';

class App extends Component {
    state = {
        user: null
    };

    render() {
        return (
            <Layout>
                <Auth />
            </Layout>
        );
    }
}

export default App;